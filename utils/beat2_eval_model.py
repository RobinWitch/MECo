import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
import argparse
from collections import OrderedDict
from scipy import linalg
import pandas as pd
import librosa
from scipy.signal import argrelextrema
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure


class L1div(object):
    def __init__(self):
        self.counter = 0
        self.sum = 0
    def run(self, results):
        self.counter += results.shape[0]
        mean = np.mean(results, 0)
        for i in range(results.shape[0]):
            results[i, :] = abs(results[i, :] - mean)
        sum_l1 = np.sum(results)
        self.sum += sum_l1
    def avg(self):
        return self.sum/self.counter
    def reset(self):
        self.counter = 0
        self.sum = 0
        
        
class FIDCalculator(object):
    '''
    todo
    '''
    def __init__(self):
        self.gt_rot = None # pandas dataframe for n frames * joints * 6
        self.gt_pos = None # n frames * (joints + 13) * 3
        self.op_rot = None # pandas dataframe for n frames * joints * 6
        self.op_pos = None # n frames * (joints + 13) * 3
        
    def _joint_selector(self, selected_joints, ori_data):
        selected_data = pd.DataFrame(columns=[])

        for joint_name in selected_joints:
            selected_data[joint_name] = ori_data[joint_name]
        return selected_data.to_numpy()
    
    
    def cal_vol(self, dtype):
        if dtype == 'pos':
            gt = self.gt_pos
            op = self.op_pos
        else:
            gt = self.gt_rot
            op = self.op_rot
        
        gt_v = gt.to_numpy()[1:, :] - gt.to_numpy()[0:-1, :]
        op_v = op.to_numpy()[1:, :] - op.to_numpy()[0:-1, :]
        if dtype == 'pos':
            self.gt_vol_pos = pd.DataFrame(gt_v, columns = gt.columns.tolist())
            self.op_vol_pos = pd.DataFrame(op_v, columns = gt.columns.tolist())
        else:
            self.gt_vol_rot = pd.DataFrame(gt_v, columns = gt.columns.tolist())
            self.op_vol_rot = pd.DataFrame(op_v, columns = gt.columns.tolist())


    @staticmethod
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


    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                    inception net (like returned by the function 'get_predictions')
                    for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                    representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                    representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        #print(mu1[0], mu2[0])
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        #print(sigma1[0], sigma2[0])
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        #print(diff, covmean[0])
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                    'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    
    def calculate_fid(self, cal_type, joint_type, high_level_opt):
        
        if cal_type == 'pos':
            if self.gt_pos.shape != self.op_pos.shape:
                min_val = min(self.gt_pos.shape[0],self.op_pos.shape[0])
                gt = self.gt_pos[:min_val]
                op = self.op_pos[:min_val]
            else:
                gt = self.gt_pos
                op = self.op_pos
            full_body = gt.columns.tolist()
        elif cal_type == 'rot':
            if self.gt_rot.shape != self.op_rot.shape:
                min_val = min(self.gt_rot.shape[0],self.op_rot.shape[0])
                gt = self.gt_rot[:min_val]
                op = self.op_rot[:min_val]
            else:
                gt = self.gt_rot
                op = self.op_rot
            full_body_with_offset = gt.columns.tolist()
            full_body = [o for o in full_body_with_offset if ('position' not in o)]
        elif cal_type == 'pos_vol':
            assert self.gt_vol_pos.shape == self.op_vol_pos.shape
            gt = self.gt_vol_pos
            op = self.op_vol_pos
            full_body_with_offset = gt.columns.tolist()
            full_body = gt.columns.tolist()
        elif cal_type == 'rot_vol':
            assert self.gt_vol_rot.shape == self.op_vol_rot.shape
            gt = self.gt_vol_rot
            op = self.op_vol_rot
            full_body_with_offset = gt.columns.tolist()
            full_body = [o for o in full_body_with_offset if ('position' not in o)]       
        #print(f'full_body contains {len(full_body)//3} joints')

        if joint_type == 'full_upper_body':
            selected_body = [o for o in full_body if ('Leg' not in o) and ('Foot' not in o) and ('Toe' not in o)] 
        elif joint_type == 'upper_body':
            selected_body = [o for o in full_body if ('Hand' not in o) and ('Leg' not in o) and ('Foot' not in o) and ('Toe' not in o)]
        elif joint_type == 'fingers':
            selected_body = [o for o in full_body if ('Hand' in o)]
        elif joint_type == 'indivdual':
            pass
        else: print('error, plz select correct joint type')
        #print(f'calculate fid for {len(selected_body)//3} joints')

        gt = self._joint_selector(selected_body, gt)
        op = self._joint_selector(selected_body, op)

        if high_level_opt == 'fid':
            fid = FIDCalculator.frechet_distance(gt, op)
            return fid
        elif high_level_opt == 'var':
            var_gt = gt.var()
            var_op = op.var()
            return var_gt, var_op
        elif high_level_opt == 'mean':
            mean_gt = gt.mean()
            mean_op = op.mean()
            return mean_gt, mean_op
        else: return 0



def get_fid_model():
    
    ckpt_path = './dataset/BEAT2/beat_english_v2.0.0/weights/AESKConv_240_100.bin'
    
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    
    args.vae_length = 240
    args.vae_layer=4
    args.vae_test_dim=330
    args.variational = False
    args.vae_grow=[1, 1, 2, 1]
    model = VAESKConv(args).cuda()
    states = torch.load(ckpt_path)
    new_weights = OrderedDict()
    flag=False
    for k, v in states['model_state'].items():
        #print(k)
        if "module" not in k:
            break
        else:
            new_weights[k[7:]]=v


    model.load_state_dict(new_weights)
    model.eval()
    return model


class VAEConv(nn.Module):
    def __init__(self, args):
        super(VAEConv, self).__init__()
        self.encoder = VQEncoderV3(args)
        self.decoder = VQDecoderV3(args)
        self.fc_mu = nn.Linear(args.vae_length, args.vae_length)
        self.fc_logvar = nn.Linear(args.vae_length, args.vae_length)
        self.variational = args.variational
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        mu, logvar = None, None
        if self.variational:
            mu = self.fc_mu(pre_latent)
            logvar = self.fc_logvar(pre_latent)
            pre_latent = reparameterize(mu, logvar)
        rec_pose = self.decoder(pre_latent)
        return {
            "poses_feat":pre_latent,
            "rec_pose": rec_pose,
            "pose_mu": mu,
            "pose_logvar": logvar,
            }
    
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        if self.variational:
            mu = self.fc_mu(pre_latent)
            logvar = self.fc_logvar(pre_latent)
            pre_latent = reparameterize(mu, logvar)
        return pre_latent
    
    def decode(self, pre_latent):
        rec_pose = self.decoder(pre_latent)
        return rec_pose

class VAESKConv(VAEConv):
    def __init__(self, args):
        super(VAESKConv, self).__init__(args)
        smpl_fname = './dataset/hub/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz'
        smpl_data = np.load(smpl_fname, encoding='latin1')
        parents = smpl_data['kintree_table'][0].astype(np.int32)
        edges = build_edge_topology(parents)
        self.encoder = LocalEncoder(args, edges)
        self.decoder = VQDecoderV3(args)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class VAESKConv(VAEConv):
    def __init__(self, args):
        super(VAESKConv, self).__init__(args)
        smpl_fname = './dataset/hub/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz'
        smpl_data = np.load(smpl_fname, encoding='latin1')
        parents = smpl_data['kintree_table'][0].astype(np.int32)
        edges = build_edge_topology(parents)
        self.encoder = LocalEncoder(args, edges)
        self.decoder = VQDecoderV3(args)


class LocalEncoder(nn.Module):
    def __init__(self, args, topology):
        super(LocalEncoder, self).__init__()
        args.channel_base = 6
        args.activation = "tanh"
        args.use_residual_blocks=True
        args.z_dim=1024
        args.temporal_scale=8
        args.kernel_size=4
        args.num_layers=args.vae_layer
        args.skeleton_dist=2
        args.extra_conv=0
        # check how to reflect in 1d
        args.padding_mode="constant"
        args.skeleton_pool="mean"
        args.upsampling="linear"


        self.topologies = [topology]
        self.channel_base = [args.channel_base]

        self.channel_list = []
        self.edge_num = [len(topology)]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        # self.convs = []

        kernel_size = args.kernel_size
        kernel_even = False if kernel_size % 2 else True
        padding = (kernel_size - 1) // 2
        bias = True
        self.grow = args.vae_grow
        for i in range(args.num_layers):
            self.channel_base.append(self.channel_base[-1]*self.grow[i])

        for i in range(args.num_layers):
            seq = []
            neighbour_list = find_neighbor(self.topologies[i], args.skeleton_dist)
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i + 1] * self.edge_num[i]
            if i == 0:
                self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)
            last_pool = True if i == args.num_layers - 1 else False

            # (T, J, D) => (T, J', D)
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbour_list), last_pool=last_pool)

            if args.use_residual_blocks:
                # (T, J, D) => (T/2, J', 2D)
                seq.append(SkeletonResidual(self.topologies[i], neighbour_list, joint_num=self.edge_num[i], in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=2, padding=padding, padding_mode=args.padding_mode, bias=bias,
                                            extra_conv=args.extra_conv, pooling_mode=args.skeleton_pool, activation=args.activation, last_pool=last_pool))
            else:
                for _ in range(args.extra_conv):
                    # (T, J, D) => (T, J, D)
                    seq.append(SkeletonConv(neighbour_list, in_channels=in_channels, out_channels=in_channels,
                                            joint_num=self.edge_num[i], kernel_size=kernel_size - 1 if kernel_even else kernel_size,
                                            stride=1,
                                            padding=padding, padding_mode=args.padding_mode, bias=bias))
                    seq.append(nn.PReLU() if args.activation == 'relu' else nn.Tanh())
                # (T, J, D) => (T/2, J, 2D)
                seq.append(SkeletonConv(neighbour_list, in_channels=in_channels, out_channels=out_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                        padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=False,
                                        in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
                # self.convs.append(seq[-1])

                seq.append(pool)
                seq.append(nn.PReLU() if args.activation == 'relu' else nn.Tanh())
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]))

        # in_features = self.channel_base[-1] * len(self.pooling_list[-1])
        # in_features *= int(args.temporal_scale / 2) 
        # self.reduce = nn.Linear(in_features, args.z_dim)
        # self.mu = nn.Linear(in_features, args.z_dim)
        # self.logvar = nn.Linear(in_features, args.z_dim)

    def forward(self, input):
        #bs, n, c = input.shape[0], input.shape[1], input.shape[2]
        output = input.permute(0, 2, 1)#input.reshape(bs, n, -1, 6)
        for layer in self.layers:
            output = layer(output)
        #output = output.view(output.shape[0], -1)
        output = output.permute(0, 2, 1)
        return output



class SkeletonConv(nn.Module):
    def __init__(self, neighbour_list, in_channels, out_channels, kernel_size, joint_num, stride=1, padding=0,
                 bias=True, padding_mode='zeros', add_offset=False, in_offset_channel=0):
        self.in_channels_per_joint = in_channels // joint_num
        self.out_channels_per_joint = out_channels // joint_num
        if in_channels % joint_num != 0 or out_channels % joint_num != 0:
            raise Exception('BAD')
        super(SkeletonConv, self).__init__()

        if padding_mode == 'zeros':
            padding_mode = 'constant'
        if padding_mode == 'reflection':
            padding_mode = 'reflect'

        self.expanded_neighbour_list = []
        self.expanded_neighbour_list_offset = []
        self.neighbour_list = neighbour_list
        self.add_offset = add_offset
        self.joint_num = joint_num

        self.stride = stride
        self.dilation = 1
        self.groups = 1
        self.padding = padding
        self.padding_mode = padding_mode
        self._padding_repeated_twice = (padding, padding)

        for neighbour in neighbour_list:
            expanded = []
            for k in neighbour:
                for i in range(self.in_channels_per_joint):
                    expanded.append(k * self.in_channels_per_joint + i)
            self.expanded_neighbour_list.append(expanded)

        if self.add_offset:
            self.offset_enc = SkeletonLinear(neighbour_list, in_offset_channel * len(neighbour_list), out_channels)

            for neighbour in neighbour_list:
                expanded = []
                for k in neighbour:
                    for i in range(add_offset):
                        expanded.append(k * in_offset_channel + i)
                self.expanded_neighbour_list_offset.append(expanded)

        self.weight = torch.zeros(out_channels, in_channels, kernel_size)
        if bias:
            self.bias = torch.zeros(out_channels)
        else:
            self.register_parameter('bias', None)

        self.mask = torch.zeros_like(self.weight)
        for i, neighbour in enumerate(self.expanded_neighbour_list):
            self.mask[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1), neighbour, ...] = 1
        self.mask = nn.Parameter(self.mask, requires_grad=False)

        self.description = 'SkeletonConv(in_channels_per_armature={}, out_channels_per_armature={}, kernel_size={}, ' \
                           'joint_num={}, stride={}, padding={}, bias={})'.format(
                               in_channels // joint_num, out_channels // joint_num, kernel_size, joint_num, stride, padding, bias
                           )

        self.reset_parameters()

    def reset_parameters(self):
        for i, neighbour in enumerate(self.expanded_neighbour_list):
            """ Use temporary variable to avoid assign to copy of slice, which might lead to unexpected result """
            tmp = torch.zeros_like(self.weight[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1),
                                   neighbour, ...])
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1),
                        neighbour, ...] = tmp
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1), neighbour, ...])
                bound = 1 / math.sqrt(fan_in)
                tmp = torch.zeros_like(
                    self.bias[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1)])
                nn.init.uniform_(tmp, -bound, bound)
                self.bias[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1)] = tmp

        self.weight = nn.Parameter(self.weight)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)

    def set_offset(self, offset):
        if not self.add_offset:
            raise Exception('Wrong Combination of Parameters')
        self.offset = offset.reshape(offset.shape[0], -1)

    def forward(self, input):
        # print('SkeletonConv')
        weight_masked = self.weight * self.mask
        #print(f'input: {input.size()}')
        res = F.conv1d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                       weight_masked, self.bias, self.stride,
                       0, self.dilation, self.groups)

        if self.add_offset:
            offset_res = self.offset_enc(self.offset)
            offset_res = offset_res.reshape(offset_res.shape + (1, ))
            res += offset_res / 100
        #print(f'res: {res.size()}')
        return res


class SkeletonLinear(nn.Module):
    def __init__(self, neighbour_list, in_channels, out_channels, extra_dim1=False):
        super(SkeletonLinear, self).__init__()
        self.neighbour_list = neighbour_list
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels_per_joint = in_channels // len(neighbour_list)
        self.out_channels_per_joint = out_channels // len(neighbour_list)
        self.extra_dim1 = extra_dim1
        self.expanded_neighbour_list = []

        for neighbour in neighbour_list:
            expanded = []
            for k in neighbour:
                for i in range(self.in_channels_per_joint):
                    expanded.append(k * self.in_channels_per_joint + i)
            self.expanded_neighbour_list.append(expanded)

        self.weight = torch.zeros(out_channels, in_channels)
        self.mask = torch.zeros(out_channels, in_channels)
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for i, neighbour in enumerate(self.expanded_neighbour_list):
            tmp = torch.zeros_like(
                self.weight[i*self.out_channels_per_joint: (i + 1)*self.out_channels_per_joint, neighbour]
            )
            self.mask[i*self.out_channels_per_joint: (i + 1)*self.out_channels_per_joint, neighbour] = 1
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[i*self.out_channels_per_joint: (i + 1)*self.out_channels_per_joint, neighbour] = tmp

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.weight = nn.Parameter(self.weight)
        self.mask = nn.Parameter(self.mask, requires_grad=False)

    def forward(self, input):
        input = input.reshape(input.shape[0], -1)
        weight_masked = self.weight * self.mask
        res = F.linear(input, weight_masked, self.bias)
        if self.extra_dim1:
            res = res.reshape(res.shape + (1,))
        return res


class SkeletonPool(nn.Module):
    def __init__(self, edges, pooling_mode, channels_per_edge, last_pool=False):
        super(SkeletonPool, self).__init__()

        if pooling_mode != 'mean':
            raise Exception('Unimplemented pooling mode in matrix_implementation')

        self.channels_per_edge = channels_per_edge
        self.pooling_mode = pooling_mode
        self.edge_num = len(edges)
        # self.edge_num = len(edges) + 1
        self.seq_list = []
        self.pooling_list = []
        self.new_edges = []
        degree = [0] * 100  # each element represents the degree of the corresponding joint

        for edge in edges:
            degree[edge[0]] += 1
            degree[edge[1]] += 1

        # seq_list contains multiple sub-lists where each sub-list is an edge chain from the joint whose degree > 2 to the end effectors or joints whose degree > 2.
        def find_seq(j, seq):
            nonlocal self, degree, edges

            if degree[j] > 2 and j != 0:
                self.seq_list.append(seq)
                seq = []

            if degree[j] == 1:
                self.seq_list.append(seq)
                return

            for idx, edge in enumerate(edges):
                if edge[0] == j:
                    find_seq(edge[1], seq + [idx])

        find_seq(0, [])
        # print(f'self.seq_list: {self.seq_list}')

        for seq in self.seq_list:
            if last_pool:
                self.pooling_list.append(seq)
                continue
            if len(seq) % 2 == 1:
                self.pooling_list.append([seq[0]])
                self.new_edges.append(edges[seq[0]])
                seq = seq[1:]
            for i in range(0, len(seq), 2):
                self.pooling_list.append([seq[i], seq[i + 1]])
                self.new_edges.append([edges[seq[i]][0], edges[seq[i + 1]][1]])
        # print(f'self.pooling_list: {self.pooling_list}')
        # print(f'self.new_egdes: {self.new_edges}')

        # add global position
        # self.pooling_list.append([self.edge_num - 1])

        self.description = 'SkeletonPool(in_edge_num={}, out_edge_num={})'.format(
            len(edges), len(self.pooling_list)
        )

        self.weight = torch.zeros(len(self.pooling_list) * channels_per_edge, self.edge_num * channels_per_edge)

        for i, pair in enumerate(self.pooling_list):
            for j in pair:
                for c in range(channels_per_edge):
                    self.weight[i * channels_per_edge + c, j * channels_per_edge + c] = 1.0 / len(pair)

        self.weight = nn.Parameter(self.weight, requires_grad=False)

    def forward(self, input: torch.Tensor):
        # print('SkeletonPool')
        # print(f'input: {input.size()}')
        # print(f'self.weight: {self.weight.size()}')
        return torch.matmul(self.weight, input)



class SkeletonResidual(nn.Module):
    def __init__(self, topology, neighbour_list, joint_num, in_channels, out_channels, kernel_size, stride, padding, padding_mode, bias, extra_conv, pooling_mode, activation, last_pool):
        super(SkeletonResidual, self).__init__()

        kernel_even = False if kernel_size % 2 else True

        seq = []
        for _ in range(extra_conv):
            # (T, J, D) => (T, J, D)
            seq.append(SkeletonConv(neighbour_list, in_channels=in_channels, out_channels=in_channels,
                                    joint_num=joint_num, kernel_size=kernel_size - 1 if kernel_even else kernel_size,
                                    stride=1,
                                    padding=padding, padding_mode=padding_mode, bias=bias))
            seq.append(nn.PReLU() if activation == 'relu' else nn.Tanh())
        # (T, J, D) => (T/2, J, 2D)
        seq.append(SkeletonConv(neighbour_list, in_channels=in_channels, out_channels=out_channels,
                                joint_num=joint_num, kernel_size=kernel_size, stride=stride,
                                padding=padding, padding_mode=padding_mode, bias=bias, add_offset=False))
        seq.append(nn.GroupNorm(10, out_channels))  # FIXME: REMEMBER TO CHANGE BACK !!!
        self.residual = nn.Sequential(*seq)

        # (T, J, D) => (T/2, J, 2D)
        self.shortcut = SkeletonConv(neighbour_list, in_channels=in_channels, out_channels=out_channels,
                                     joint_num=joint_num, kernel_size=1, stride=stride, padding=0,
                                     bias=True, add_offset=False)

        seq = []
        # (T/2, J, 2D) => (T/2, J', 2D)
        pool = SkeletonPool(edges=topology, pooling_mode=pooling_mode,
                            channels_per_edge=out_channels // len(neighbour_list), last_pool=last_pool)
        if len(pool.pooling_list) != pool.edge_num:
            seq.append(pool)
        seq.append(nn.PReLU() if activation == 'relu' else nn.Tanh())
        self.common = nn.Sequential(*seq)

    def forward(self, input):
        output = self.residual(input) + self.shortcut(input)

        return self.common(output)




def find_neighbor(edges, d):
    """
    Args:
        edges: The list contains N elements, each element represents (parent, child).
        d: Distance between edges (the distance of the same edge is 0 and the distance of adjacent edges is 1).

    Returns:
        The list contains N elements, each element is a list of edge indices whose distance <= d.
    """
    edge_mat = calc_edge_mat(edges)
    neighbor_list = []
    edge_num = len(edge_mat)
    for i in range(edge_num):
        neighbor = []
        for j in range(edge_num):
            if edge_mat[i][j] <= d:
                neighbor.append(j)
        neighbor_list.append(neighbor)

    # # add neighbor for global part
    # global_part_neighbor = neighbor_list[0].copy()
    # """
    # Line #373 is buggy. Thanks @crissallan!!
    # See issue #30 (https://github.com/DeepMotionEditing/deep-motion-editing/issues/30)
    # However, fixing this bug will make it unable to load the pretrained model and
    # affect the reproducibility of quantitative error reported in the paper.
    # It is not a fatal bug so we didn't touch it and we are looking for possible solutions.
    # """
    # for i in global_part_neighbor:
    #     neighbor_list[i].append(edge_num)
    # neighbor_list.append(global_part_neighbor)

    return neighbor_list

def build_edge_topology(topology):
    # get all edges (pa, child)
    edges = []
    joint_num = len(topology)
    edges.append((0, joint_num))  # add an edge between the root joint and a virtual joint
    for i in range(1, joint_num):
        edges.append((topology[i], i))
    return edges

def calc_edge_mat(edges):
    edge_num = len(edges)
    # edge_mat[i][j] = distance between edge(i) and edge(j)
    edge_mat = [[100000] * edge_num for _ in range(edge_num)]
    for i in range(edge_num):
        edge_mat[i][i] = 0

    # initialize edge_mat with direct neighbor
    for i, a in enumerate(edges):
        for j, b in enumerate(edges):
            link = 0
            for x in range(2):
                for y in range(2):
                    if a[x] == b[y]:
                        link = 1
            if link:
                edge_mat[i][j] = 1

    # calculate all the pairs distance
    for k in range(edge_num):
        for i in range(edge_num):
            for j in range(edge_num):
                edge_mat[i][j] = min(edge_mat[i][j], edge_mat[i][k] + edge_mat[k][j])
    return edge_mat


class VQDecoderV3(nn.Module):
    def __init__(self, args):
        super(VQDecoderV3, self).__init__()
        n_up = args.vae_layer
        channels = []
        for i in range(n_up-1):
            channels.append(args.vae_length)
        channels.append(args.vae_length)
        channels.append(args.vae_test_dim)
        input_size = args.vae_length
        n_resblk = 2
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs
        
class VQEncoderV3(nn.Module):
    def __init__(self, args):
        super(VQEncoderV3, self).__init__()
        n_down = args.vae_layer
        channels = [args.vae_length]
        for i in range(n_down-1):
            channels.append(args.vae_length)
        
        input_size = args.vae_test_dim
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out
    




class alignment(object):
    def __init__(self, sigma, order, mmae=None, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21]):
        self.sigma = sigma
        self.order = order
        self.upper_body= upper_body
        # self.times = self.oenv = self.S = self.rms = None
        self.pose_data = []
        self.mmae = mmae
        self.threshold = 0.3
    
    def load_audio(self, wave, t_start=None, t_end=None, without_file=False, sr_audio=16000):
        hop_length = 512
        if without_file:
            y = wave
            sr = sr_audio
        else: y, sr = librosa.load(wave)
        if t_start is None:
            short_y = y
        else:
            short_y = y[t_start:t_end]
        # print(short_y.shape)
        onset_t = librosa.onset.onset_detect(y=short_y, sr=sr_audio, hop_length=hop_length, units='time')
        return onset_t

    def load_pose(self, pose, t_start, t_end, pose_fps, without_file=False):
        data_each_file = []
        if without_file:
            for line_data_np in pose: #,args.pre_frames, args.pose_length
                data_each_file.append(line_data_np)
                    #data_each_file.append(np.concatenate([line_data_np[9:18], line_data_np[75:84], ],0))
        else: 
            with open(pose, "r") as f:
                for i, line_data in enumerate(f.readlines()):
                    if i < 432: continue
                    line_data_np = np.fromstring(line_data, sep=" ",)
                    if pose_fps == 15:
                        if i % 2 == 0:
                            continue
                    data_each_file.append(np.concatenate([line_data_np[30:39], line_data_np[112:121], ],0))
                    
        data_each_file = np.array(data_each_file)
        #print(data_each_file.shape)
        
        joints = data_each_file.transpose(1, 0)
        dt = 1/pose_fps
        # first steps is forward diff (t+1 - t) / dt
        init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
        # middle steps are second order (t+1 - t-1) / 2dt
        middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
        # last step is backward diff (t - t-1) / dt
        final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
        #print(joints.shape, init_vel.shape, middle_vel.shape, final_vel.shape)
        vel = np.concatenate([init_vel, middle_vel, final_vel], 1).transpose(1, 0).reshape(data_each_file.shape[0], -1, 3)
        #print(vel.shape)
        #vel = data_each_file.reshape(data_each_file.shape[0], -1, 3)[1:] - data_each_file.reshape(data_each_file.shape[0], -1, 3)[:-1]
        vel = np.linalg.norm(vel, axis=2) / self.mmae
        
        beat_vel_all = []
        for i in range(vel.shape[1]):
            vel_mask = np.where(vel[:, i]>self.threshold)
            #print(vel.shape)
            #t_end = 80
            #vel[::2, :] -= 0.000001
            #print(vel[t_start:t_end, i], vel[t_start:t_end, i].shape)
            beat_vel = argrelextrema(vel[t_start:t_end, i], np.less, order=self.order) # n*47
            #print(beat_vel, t_start, t_end)
            beat_vel_list = []
            for j in beat_vel[0]:
                if j in vel_mask[0]:
                    beat_vel_list.append(j)
            beat_vel = np.array(beat_vel_list)
            beat_vel_all.append(beat_vel)
        #print(beat_vel_all)
        return beat_vel_all #beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist
    
    
    def load_data(self, wave, pose, t_start, t_end, pose_fps):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.load_pose(pose, t_start, t_end, pose_fps)
        return onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist 

    def eval_random_pose(self, wave, pose, t_start, t_end, pose_fps, num_random=60):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        dur = t_end - t_start
        for i in range(num_random):
            beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.load_pose(pose, i, i+dur, pose_fps)
            dis_all_b2a= self.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist)
            print(f"{i}s: ",dis_all_b2a)


    @staticmethod
    def plot_onsets(audio, sr, onset_times_1, onset_times_2):
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt
        # Plot audio waveform
        fig, axarr = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot audio waveform in both subplots
        librosa.display.waveshow(audio, sr=sr, alpha=0.7, ax=axarr[0])
        librosa.display.waveshow(audio, sr=sr, alpha=0.7, ax=axarr[1])
        
        # Plot onsets from first method on the first subplot
        for onset in onset_times_1:
            axarr[0].axvline(onset, color='r', linestyle='--', alpha=0.9, label='Onset Method 1')
        axarr[0].legend()
        axarr[0].set(title='Onset Method 1', xlabel='', ylabel='Amplitude')
        
        # Plot onsets from second method on the second subplot
        for onset in onset_times_2:
            axarr[1].axvline(onset, color='b', linestyle='-', alpha=0.7, label='Onset Method 2')
        axarr[1].legend()
        axarr[1].set(title='Onset Method 2', xlabel='Time (s)', ylabel='Amplitude')
    
        
        # Add legend (eliminate duplicate labels)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        # Show plot
        plt.title("Audio waveform with Onsets")
        plt.savefig("./onset.png", dpi=500)
    
    def audio_beat_vis(self, onset_raw, onset_bt, onset_bt_rms):
        figure(figsize=(24, 6), dpi=80)
        fig, ax = plt.subplots(nrows=4, sharex=True)
        librosa.display.specshow(librosa.amplitude_to_db(self.S, ref=np.max),
                                y_axis='log', x_axis='time', ax=ax[0])
        ax[0].label_outer()
        ax[1].plot(self.times, self.oenv, label='Onset strength')
        ax[1].vlines(librosa.frames_to_time(onset_raw), 0, self.oenv.max(), label='Raw onsets', color='r')
        ax[1].legend()
        ax[1].label_outer()

        ax[2].plot(self.times, self.oenv, label='Onset strength')
        ax[2].vlines(librosa.frames_to_time(onset_bt), 0, self.oenv.max(), label='Backtracked', color='r')
        ax[2].legend()
        ax[2].label_outer()

        ax[3].plot(self.times, self.rms[0], label='RMS')
        ax[3].vlines(librosa.frames_to_time(onset_bt_rms), 0, self.oenv.max(), label='Backtracked (RMS)', color='r')
        ax[3].legend()
        fig.savefig("./onset.png", dpi=500)
    
    @staticmethod
    def motion_frames2time(vel, offset, pose_fps):
        time_vel = vel/pose_fps + offset 
        return time_vel    
    
    @staticmethod
    def GAHR(a, b, sigma):
        dis_all_a2b = 0
        dis_all_b2a = 0
        for b_each in b:
            l2_min = np.inf
            for a_each in a:
                l2_dis = abs(a_each - b_each)
                if l2_dis < l2_min:
                    l2_min = l2_dis
            dis_all_b2a += math.exp(-(l2_min**2)/(2*sigma**2))
        dis_all_b2a /= len(b)
        return dis_all_b2a 
    
    @staticmethod
    def fix_directed_GAHR(a, b, sigma):
        a = alignment.motion_frames2time(a, 0, 30)
        b = alignment.motion_frames2time(b, 0, 30)
        t = len(a)/30
        a = [0] + a + [t]
        b = [0] + b + [t]
        dis_a2b = alignment.GAHR(a, b, sigma)
        return dis_a2b

    def calculate_align(self, onset_bt_rms, beat_vel, pose_fps=30):
        audio_bt = onset_bt_rms
        avg_dis_all_b2a_list = []
        for its, beat_vel_each in enumerate(beat_vel):
            if its not in self.upper_body:
                continue
            #print(beat_vel_each)
            #print(audio_bt.shape, beat_vel_each.shape)
            pose_bt = self.motion_frames2time(beat_vel_each, 0, pose_fps)
            #print(pose_bt)
            avg_dis_all_b2a_list.append(self.GAHR(pose_bt, audio_bt, self.sigma))
        # avg_dis_all_b2a = max(avg_dis_all_b2a_list)
        avg_dis_all_b2a = sum(avg_dis_all_b2a_list)/len(avg_dis_all_b2a_list) #max(avg_dis_all_b2a_list)
        #print(avg_dis_all_b2a, sum(avg_dis_all_b2a_list)/47)
        return avg_dis_all_b2a  
    
    
if __name__ == "__main__":
    model = get_fid_model()
    a = model