import sys
sys.path.append('./refer/fairseq')
import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datasets import Dataset, DatasetDict
import numpy as np
from huggingface_hub import hf_hub_download
from typing import List, Tuple
import torch
import torchaudio
import sys
sys.path.append('./')
from models.vq.model import get_rvqvae_model
from utils import rotation_conversions as rc


import logging
import os
import sys
import joblib
import fairseq
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from torchaudio.functional import resample

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000, fp16=False, sampling_rate=16000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().to(DEVICE)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.fp16 = fp16
        if fp16:
            self.model.half()
        
        self.layer_shift = 0
        self.target_sample_hz = sampling_rate

    def read_audio(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.target_sample_hz:
            wav = resample(wav, sr, self.target_sample_hz)
        return wav

    @torch.no_grad()
    def get_feats(self, waveform):
        x = waveform
        with torch.no_grad():
            if self.fp16:
                x = x.half().cuda()
            else:
                x = x.float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                        source=x_chunk,
                        padding_mask=None,
                        mask=False,
                        output_layer=self.layer + self.layer_shift,
                )
        
                feat.append(feat_chunk)
        if len(feat) == 0:
            return torch.zeros(0, 0)
        return torch.cat(feat, 1).squeeze(0)

class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            self.C = self.C.to(x)
            self.Cnorm = self.Cnorm.to(x)
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


class Speech2Unit(torch.nn.Module):
    def __init__(
        self, 
        ckpt_dir,
        layer=11, 
        max_chunk=1600000, 
        fp16=False, 
        sampling_rate=16000,
        ):

        """
        Args:
            ckpt_dir(str): path to hubert model dir(e.g. hubert_base_ls960.pt)
            layer(int): feat from which layer of hubert models defauly by 9
            max_chunk(int): default by 1600000
            fp16(bool): default by False
            sampling_rate(int): sampling_rate default by 16000
        """
        super().__init__()

        ckpt_path = os.path.join(ckpt_dir, "mhubert_base_vp_en_es_fr_it3.pt")
        km_path = os.path.join(ckpt_dir, "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")

        self.feature_reader = FeatureReader(ckpt_path, layer, max_chunk, fp16, sampling_rate)
        self.apply_kmeans = ApplyKmeans(km_path)
    
    @staticmethod
    def merge_duplicates(cluster_ids):
        dup_cluster_list = []
        duration_list = []
        count = 1
        for i in range(0, len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i+1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                duration_list.append(count)
                count = 1
        return dup_cluster_list, duration_list
    

    def __call__(self, path, merged=False):
        waveform = self.feature_reader.read_audio(path).to(DEVICE)
        
        feat = self.feature_reader.get_feats(waveform)
        cluster_ids = self.apply_kmeans(feat).tolist()
        dup_cluster_list, duration_list = self.merge_duplicates(cluster_ids)

        merged_units = "<sosp>" + "".join([f"<{str(x)}>" for x in dup_cluster_list]) + "<eosp>"
        unmerged_units = " ".join([f"{str(x)}" for x in cluster_ids])

        if merged:
            return merged_units
        else:
            return unmerged_units

    def encode(self, wav,sr, merged=False):
        waveform = wav.to(DEVICE)
        
        feat = self.feature_reader.get_feats(waveform)
        cluster_ids = self.apply_kmeans(feat).tolist()
        return cluster_ids



ckpt_dir = "./ckpt/mhubert_base_1000"


pose_norm = True
pose_mean = np.load('./mean_std/beat2_2_330_mean.npy')
pose_std = np.load('./mean_std/beat2_2_330_std.npy')

pose_trans_mean = np.load('./mean_std/beat2_2_trans_mean.npy')
pose_trans_std = np.load('./mean_std/beat2_2_trans_std.npy')

audio_tokenizer = Speech2Unit(ckpt_dir=ckpt_dir)
motion_vq_ckpt_path = "./ckpt/MECo_BEAT2_2_RVQVAE/net_300000.pth"
motion_tokenizer = get_rvqvae_model(motion_vq_ckpt_path).cuda()

wav_dir_path = "./dataset/BEAT2/beat_english_v2.0.0/wave16k"
pose_dir_path = "./dataset/BEAT2/beat_english_v2.0.0/smplxflame_30"

motion_squeeze_time = 4 # 30fps motion frames encode to 30/4 motion codes

key_id_list = []
audio_tokens_list = []
motion_tokens_list = []

for i in range(1,119):
    wav_file_path = wav_dir_path+f'/2_scott_0_{i}_{i}.wav'
    pose_file_path = pose_dir_path+f'/2_scott_0_{i}_{i}.npz'
    wav, sr = torchaudio.load(wav_file_path)

    
    pose_data = np.load(pose_file_path, allow_pickle=True)
    pose_each_file = pose_data["poses"]
    trans_each_file = pose_data["trans"]
    
    
    cut_off_length = 0
    audio_length = wav[0].shape[0]/16000
    motion_length = (pose_each_file.shape[0]-pose_each_file.shape[0]%motion_squeeze_time)/30
    
    if audio_length > motion_length:
        print("audio longer")
        cut_off_length = motion_length
    else:  
        print("motion longer")
        cut_off_length = audio_length
    
    wav = wav[:, :int(cut_off_length*16000)]

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
    
    tar_pose = tar_pose[:int(cut_off_length*30),mask].cuda()


    sub_seg_num = cut_off_length/16
    sub_seg_num = int(sub_seg_num)
    for j in range(sub_seg_num):
        key_id = f"2_scott_0_{i}_{i}_{j}"
        sub_wav = wav[:, j*16*16000:(j+1)*16*16000+80]
        audio_tokens = audio_tokenizer.encode(sub_wav,sr)
        assert len(audio_tokens) == 800, "error"
        motion_tokens = motion_tokenizer.encode(tar_pose[j*16*30:(j+1)*16*30].unsqueeze(0))[0].squeeze().cpu().tolist()
        key_id_list.append(key_id)
        audio_tokens_list.append(audio_tokens)
        motion_tokens_list.append(motion_tokens)
    if (cut_off_length - (j+1)*16) > 4:
        key_id = f"2_scott_0_{i}_{i}_{j+1}"
        sub_wav = wav[:, (j+1)*16*16000:]
        audio_tokens = audio_tokenizer.encode(sub_wav,sr)
        motion_tokens = motion_tokenizer.encode(tar_pose[(j+1)*16*30:].unsqueeze(0))[0].squeeze().cpu().tolist()
        key_id_list.append(key_id)
        audio_tokens = audio_tokens[:(len(audio_tokens)-len(audio_tokens)%20)]
        motion_tokens = motion_tokens[:int(len(audio_tokens)*0.15)]
        
        audio_tokens_list.append(audio_tokens)
        motion_tokens_list.append(motion_tokens)

audio_tokens = [[f'<|audio_{num:04d}|>' for num in audio_tokens] for audio_tokens in audio_tokens_list]
motion_tokens = [[f'<|motion_{num:04d}|>' for num in motion_tokens] for motion_tokens in motion_tokens_list]

upload_texts = audio_tokens + motion_tokens
type_list = ["audio_token"]*len(audio_tokens) + ["motion_token"]*len(motion_tokens)
file_id_list = key_id_list*2
data_dict = {
    "file": file_id_list,
    "text": upload_texts,
    "type": type_list,
}

train_file_id_list = []
train_text_list = []
train_type_list = []

test_file_id_list = []
test_text_list = []
test_type_list = []

valid_file_id_list = []
valid_text_list = []
valid_type_list = []


test_id = [103,111,1,2,3,4,5,65,6,73,7,81,87,8,95]
valid_id = [9,10,11,12,13,14,15,16]
train_id = list(set(range(1,119)) - set(test_id) - set(valid_id))

for item in zip(data_dict['file'],data_dict['text'],data_dict['type']):
    full_name = item[0]
    main_name = full_name.split('_')[3]
    if int(main_name) in test_id:
        test_file_id_list.append(item[0])
        test_text_list.append(item[1])
        test_type_list.append(item[2])
    elif int(main_name) in train_id:
        train_file_id_list.append(item[0])
        train_text_list.append(item[1])
        train_type_list.append(item[2])
    elif int(main_name) in valid_id:
        valid_file_id_list.append(item[0])
        valid_text_list.append(item[1])
        valid_type_list.append(item[2])
    else:
        print(f"Error: {main_name}")
        

traidata_dict = {
    "file": train_file_id_list,
    "text": train_text_list,
    "type": train_type_list,
}

testdata_dict = {
    "file": test_file_id_list,
    "text": test_text_list,
    "type": test_type_list,
}

valid_dict = {
    "file": valid_file_id_list,
    "text": valid_text_list,
    "type": valid_type_list,
}

dataset = DatasetDict({
    "all_data":Dataset.from_dict(data_dict),
    "train": Dataset.from_dict(traidata_dict),
    "test": Dataset.from_dict(testdata_dict),
    "validation": Dataset.from_dict(valid_dict),
})

print(dataset)

your_huggingface_name = "robinwitch"
dataset.push_to_hub(f"{your_huggingface_name}/meco_mhubert1000_beat2_2")
print(f"Please check the dataset at https://huggingface.co/datasets/{your_huggingface_name}/meco_mhubert1000_beat2_2")
print(f"cd dataset && git clone https://huggingface.co/datasets/{your_huggingface_name}/meco_mhubert1000_beat2_2")

