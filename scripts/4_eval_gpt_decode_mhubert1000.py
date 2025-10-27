import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append('./')
from models.vq.model import get_rvqvae_model
import torch
import numpy as np
from utils.beat2_eval_model import get_fid_model,FIDCalculator,L1div,alignment
from utils import rotation_conversions as rc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import re
import smplx
import time
import librosa

model_name = f"./ckpt/MECo_BEAT2_2_qwen2.5_0.5b_stage2"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


result_save_path = "./output/visual_result/"
time_local = time.localtime()
name_expend = "%02d%02d_%02d%02d%02d/"%(time_local[1], time_local[2],time_local[3], time_local[4], time_local[5])
result_save_path = result_save_path + name_expend

if os.path.exists(result_save_path) is False:
    os.makedirs(result_save_path)


default_qwen_system_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"


class AllowedTokensLogitsProcessor:
    def __init__(self, allowed_token_ids, filter_value=-float("Inf")):
        self.allowed_token_ids = set(allowed_token_ids)
        self.filter_value = filter_value

    def __call__(self, input_ids, scores):
        # 创建一个全为过滤值的掩码
        mask = torch.full(scores.shape, self.filter_value, device=scores.device)
        # 仅将允许的 token 的 logits 设置为原值
        mask[:, list(self.allowed_token_ids)] = 0
        # 应用掩码
        scores = scores + mask
        return scores

motion_tokens = list(range(156690, 156690+512))

motion_logits_processor = LogitsProcessorList([AllowedTokensLogitsProcessor(motion_tokens)])

smplx_model = smplx.create(
            "./dataset/hub/smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).eval()

def get_gpt_generation_result(audio_tokens, motion_tokens):
    seg_num = (len(audio_tokens)-20) // 180
    motion_tokens_str = "".join(motion_tokens[:3])
    for i in range(seg_num):
        motion_seed_tokens = motion_tokens_str[-45:]    #一个motion token是15个字符，我们取3*15=45个字符作为seed
        audio_input = "".join(audio_tokens[180*i:20+180*(i+1)])
        
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": audio_input},
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        text = text[len(default_qwen_system_prompt):] + motion_seed_tokens
        
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=27,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=motion_logits_processor,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        motion_tokens_str+=response
    
    matches = re.findall(r'<\|motion_\d{4}\|>', motion_tokens_str)
    motion_tokens_list = [int(match[9:13]) for match in matches]
    return motion_tokens_list

eval_fid_model = get_fid_model()
motion_vq_ckpt_path = "./ckpt/MECo_BEAT2_2_RVQVAE/net_300000.pth"
motion_tokenizer = get_rvqvae_model(motion_vq_ckpt_path).cuda()
l1_calculator = L1div()


align_mask = 60
avg_vel = np.load("./dataset/BEAT2/beat_english_v2.0.0/weights/mean_vel_smplxflame_30.npy")
alignmenter = alignment(0.3, 7, avg_vel, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21])
align = 0
total_length = 0

pose_mean = torch.from_numpy(np.load("./mean_std/beat2_2_330_mean.npy")).float()
pose_std = torch.from_numpy(np.load("./mean_std/beat2_2_330_std.npy")).float()
pose_trans_mean = np.load("./mean_std/beat2_2_trans_mean.npy")
pose_trans_std = np.load("./mean_std/beat2_2_trans_std.npy")

latent_ori = []
latent_out = []


def load_normal_tar_pose(pose_file_path):
    # 输入是一个pose文件的路径，输出是一个处理好的pose，维度为(1, 52*6+3)，52是52个关节点，6是6维的旋转表示，3是位移
    pose_data = np.load(pose_file_path, allow_pickle=True)
    pose_each_file = pose_data["poses"]
    trans_each_file = pose_data["trans"]
    
    trans_each_file[:,0] = trans_each_file[:,0] - trans_each_file[0,0]
    trans_each_file[:,2] = trans_each_file[:,2] - trans_each_file[0,2]
    trans_v_each_file = np.zeros_like(trans_each_file)
    trans_v_each_file[1:,0] = trans_each_file[1:,0] - trans_each_file[:-1,0]
    trans_v_each_file[0,0] = trans_v_each_file[1,0]
    trans_v_each_file[1:,2] = trans_each_file[1:,2] - trans_each_file[:-1,2]
    trans_v_each_file[0,2] = trans_v_each_file[1,2]
    trans_v_each_file[:,1] = trans_each_file[:,1]
    tar_pose = torch.from_numpy(pose_each_file).float()
    tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(-1, 55, 3))
    tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(-1, 55*6)
    tar_pose = (tar_pose - pose_mean) / pose_std
    trans_v = (trans_v_each_file-pose_trans_mean)/pose_trans_std
    trans_v = torch.from_numpy(trans_v).float()
    tar_pose = torch.cat([tar_pose, trans_v], dim=1)
    mask = list(range(0,22*6))+list(range(25*6,55*6)) + list(range(55*6,55*6+3))
    tar_mask_pose = tar_pose[:,mask].unsqueeze(0).cuda()
    tar_pose = tar_pose.unsqueeze(0).cuda()
    return tar_pose, tar_mask_pose

def reverse_pose(pose):
    
    rec_trans_v = pose[...,-3:]
    rec_trans_v = rec_trans_v * pose_trans_std + pose_trans_mean
    rec_trans = torch.zeros_like(rec_trans_v)
    rec_trans = torch.cumsum(rec_trans_v, dim=-2)
    rec_trans[...,1]=rec_trans_v[...,1]
    rec_trans = rec_trans
    
    body_pose = pose[...,:-3]
    body_pose = body_pose * pose_std + pose_mean
    
    body_pose = body_pose
    return body_pose, rec_trans

def frechet_distance(samples_A, samples_B):
    A_mu = np.mean(samples_A, axis=0)
    A_sigma = np.cov(samples_A, rowvar=False)
    B_mu = np.mean(samples_B, axis=0)
    B_sigma = np.cov(samples_B, rowvar=False)
    try:
        frechet_dist = FIDCalculator.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
    except ValueError:
        frechet_dist = 1e+10
    return frechet_dist



ds = load_dataset("./dataset/meco_mhubert1000_beat2_2")['all_data']

token_list = []
last_file = None

all_motion_tokens_list = []

for item in ds:
    
    if item['type'] != 'motion_token':
        continue
    file_name = item['file'][:-2]
    if file_name!=last_file:
        if len(token_list) > 0:
            all_motion_tokens_list.append(token_list)
            
        last_file = file_name
        token_list = []
    
    tmp = item['text']
    # for i in range(len(tmp)):
    #     tmp[i] = int(tmp[i][9:13])
    token_list+=tmp

all_motion_tokens_list.append(token_list)



token_list = []
last_file = None
all_audio_tokens_list = []
for item in ds:
    if item['type'] != 'audio_token':
        continue
    file_name = item['file'][:-2]
    if file_name!=last_file:
        if len(token_list) > 0:
            all_audio_tokens_list.append(token_list)
        last_file = file_name
        token_list = []
    tmp = item['text']
    token_list+=tmp
all_audio_tokens_list.append(token_list)





wav_dir_path = "./dataset/BEAT2/beat_english_v2.0.0/wave16k"
pose_dir_path = "./dataset/BEAT2/beat_english_v2.0.0/smplxflame_30"

motion_squeeze_time = 4 # 30fps motion frames encode to 30/4 motion codes

key_id_list = []
audio_tokens_list = []
motion_tokens_list = []

test_id = [103,111,1,2,3,4,5,65,6,73,7,81,87,8,95]
train_id = list(set(range(1,119)) - set(test_id))

# test_id = list(range(20,36))
for i in test_id:
    pose_file_path = pose_dir_path+f'/2_scott_0_{i}_{i}.npz'
    normal_tar_pose, normal_tar_mask_pose = load_normal_tar_pose(pose_file_path)
    #convert_to_tokenlist = motion_tokenizer.encode(normal_tar_mask_pose)[0][0][:,0].tolist()
    convert_to_tokenlist = get_gpt_generation_result(all_audio_tokens_list[i-1], all_motion_tokens_list[i-1])
    
    
    # ######################### debug #########################
    
    # audio_tokens_int_str = [int(item[8:12]) for item in all_audio_tokens_list[i-1]]
    # print(audio_tokens_int_str)
    
    # ######################### debug #########################
    
    rec_normal_pose = motion_tokenizer.forward_decoder(torch.tensor(convert_to_tokenlist).reshape(1,-1,1).cuda())

    cutoff_length = min(normal_tar_pose.shape[1], rec_normal_pose.shape[1])
    
    remain = cutoff_length % 32 
    cutoff_length = cutoff_length - remain
    # print(f"cutoff_length: {cutoff_length}, normal_tar_pose_length: {normal_tar_pose.shape[1]}, rec_normal_pose_length: {rec_normal_pose.shape[1]}")
    rec_normal_pose = rec_normal_pose[:,:cutoff_length]
    normal_tar_pose = normal_tar_pose[:,:cutoff_length]
    
    rec_normal_pose = torch.concat([rec_normal_pose[...,:22*6],normal_tar_pose[...,22*6:25*6],rec_normal_pose[...,22*6:]],dim=-1)
    
    tar_pose,tar_trans = reverse_pose(normal_tar_pose.detach().cpu())
    rec_pose,rec_trans = reverse_pose(rec_normal_pose.detach().cpu())
    latent_out.append(eval_fid_model.map2latent(rec_pose.cuda()).squeeze().detach().cpu().numpy())
    latent_ori.append(eval_fid_model.map2latent(tar_pose.cuda()).squeeze().detach().cpu().numpy())
    
    gt_npz = np.load(pose_file_path, allow_pickle=True)
    rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(-1, 55, 6))
    rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(-1, 55*3)
    vertices_rec = smplx_model(
            betas=torch.tensor(gt_npz["betas"].reshape(-1, 300)).float(), 
            transl=rec_trans.reshape(-1, 3)-rec_trans.reshape(-1, 3), 
            expression=torch.tensor(gt_npz["expressions"].reshape(-1, 100)-gt_npz["expressions"].reshape(-1, 100))[:cutoff_length].float(),
            jaw_pose=rec_pose[:, 66:69], 
            global_orient=rec_pose[:,:3], 
            body_pose=rec_pose[:,3:21*3+3], 
            left_hand_pose=rec_pose[:,25*3:40*3], 
            right_hand_pose=rec_pose[:,40*3:55*3], 
            return_joints=True, 
            leye_pose=rec_pose[:, 69:72], 
            reye_pose=rec_pose[:, 72:75],
        )
    joints_rec = vertices_rec["joints"].detach().cpu().numpy().reshape(1, -1, 127*3)[0, :cutoff_length, :55*3]
    
    l1_calculator.run(joints_rec)
    audio_sr = 16000
    pose_fps = 30
    in_audio_eval, sr = librosa.load(f"./dataset/BEAT2/beat_english_v2.0.0/wave16k/2_scott_0_{i}_{i}.wav",sr = audio_sr)

    a_offset = int(align_mask * (audio_sr / pose_fps))
    onset_bt = alignmenter.load_audio(in_audio_eval[:int(audio_sr / pose_fps*cutoff_length)], a_offset, len(in_audio_eval)-a_offset, True)
    
    beat_vel = alignmenter.load_pose(joints_rec, align_mask, cutoff_length-align_mask, 30, True)
    align += (alignmenter.calculate_align(onset_bt, beat_vel, 30) * (cutoff_length-2*align_mask))
    total_length += cutoff_length

    save_npz = True
    if save_npz:

        rec_pose_np = rec_pose.detach().cpu().numpy()
        rec_trans_np = rec_trans.detach().cpu().numpy().reshape(-1, 3)

        np.savez(result_save_path+f'gen_2_scott_0_{i}_{i}.npz',
            betas=gt_npz["betas"],
            poses=rec_pose_np,
            expressions=gt_npz["expressions"],
            trans=rec_trans_np,
            model='smplx2020',
            gender='neutral',
            mocap_frame_rate = 30,
        )
    
latent_out_all = np.concatenate(latent_out, axis=0)
latent_ori_all = np.concatenate(latent_ori, axis=0)

fid = FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
print(f"fid: {fid}")

l1div = l1_calculator.avg()
print(f"l1div score: {l1div}")

align_avg = align/(total_length-2*len(test_id)*align_mask)
print(f"align score: {align_avg}")