import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append("./")
import json

import torch
import torch.nn as nn
import torch.optim as optim
import logging


import warnings
warnings.filterwarnings('ignore')
from models.vq.model import RVQVAE

def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


class ReConsLoss(nn.Module):
    def __init__(self, recons_loss):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
    
    def my_forward(self,motion_pred,motion_gt,mask) :
        loss = self.Loss(motion_pred[..., mask], motion_gt[..., mask])
        return loss



import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')
    parser.add_argument('--body_part',type=str,default='whole_trans')
    ## optimization
    parser.add_argument('--total-iter', default=300000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[100000, 200000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss-vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l2', help='reconstruction loss')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume-gpt", type=str, default=None, help='resume pth for GPT')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output/', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='RVQVAE', help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')
    
    
    return parser.parse_args()

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

##### ---- Exp dirs ---- #####
args = get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}_{args.body_part}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = get_logger(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


##### ---- Dataloader ---- #####
from dataloaders.mix_sep import CustomDataset
from utils.config import parse_args

dataset_args = parse_args("configs/beat2_rvqvae.yaml")
build_cache = not os.path.exists(dataset_args.cache_path)

trainSet = CustomDataset(dataset_args,"train",build_cache = build_cache)
train_loader = torch.utils.data.DataLoader(trainSet,
                                              args.batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              drop_last = True)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

train_loader_iter = cycle(train_loader)



if args.body_part in "upper":
    joints = [3,6,9,12,13,14,15,16,17,18,19,20,21]
    upper_body_mask = []
    for i in joints:
        upper_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    mask = upper_body_mask
    rec_mask = list(range(len(mask)))

    
elif args.body_part in "hands":

    joints = list(range(25,55))
    hands_body_mask = []
    for i in joints:
        hands_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    mask = hands_body_mask
    rec_mask = list(range(len(mask)))


elif args.body_part in "lower":
    joints = [0,1,2,4,5,7,8,10,11]
    lower_body_mask = []
    for i in joints:
        lower_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    mask = lower_body_mask
    rec_mask = list(range(len(mask)))

elif args.body_part in "lower_trans":
    joints = [0,1,2,4,5,7,8,10,11]
    lower_body_mask = []
    for i in joints:
        lower_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    lower_body_mask.extend([330,331,332])
    mask = lower_body_mask
    rec_mask = list(range(len(mask)))

elif args.body_part in "whole_trans":
    joints = list(range(0,22))+list(range(25,55))
    whole_body_mask = []
    for i in joints:
        whole_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    whole_body_mask.extend([330,331,332])
    mask = whole_body_mask
    rec_mask = list(range(len(mask)))



##### ---- Network ---- #####
if args.body_part in "upper":
    dim_pose = 78   
elif args.body_part in "hands":
    dim_pose = 180
elif args.body_part in "lower":
    dim_pose = 54
elif args.body_part in "lower_trans":
    dim_pose = 57
elif args.body_part in "whole":
    dim_pose = 312
elif args.body_part in "whole_trans":
    dim_pose = 315

args.num_quantizers = 6
args.shared_codebook =  False
args.quantize_dropout_prob = 0.2
net = RVQVAE(args,
            dim_pose,
            args.nb_code,
            args.code_dim,
            args.code_dim,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
            args.vq_act,
            args.vq_norm)


if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
net.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)


Loss = ReConsLoss(args.recons_loss)

##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

for nb_iter in range(1, args.warm_up_iter):
    
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    
    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion[...,mask].cuda().float() # (bs, 64, dim)

    pred_motion, loss_commit, perplexity = net(gt_motion).values()
    loss_motion = Loss.my_forward(pred_motion, gt_motion,rec_mask)
    loss_vel = 0#Loss.my_forward(pred_motion, gt_motion,vel_mask)
    
    loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

##### ---- Training ---- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
args.eval_iter = args.eval_iter * 10
for nb_iter in range(1, args.total_iter + 1):
    
    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion[...,mask].cuda().float() # bs, nb_joints, joints_dim, seq_len
    
    pred_motion, loss_commit, perplexity = net(gt_motion).values()
    loss_motion = Loss.my_forward(pred_motion, gt_motion,rec_mask)
    loss_vel = 0
    
    loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter

        logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,

    if nb_iter % args.eval_iter==0 :
        torch.save({'net' : net.state_dict()}, os.path.join(args.out_dir, f'net_{nb_iter}.pth'))

        
