import random
import torch
import torch.nn as nn
from models.vq.encdec import Encoder, Decoder, Causal_Encoder, Causal_Decoder
from models.vq.residual_vq import ResidualVQ
import numpy as np

class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def forward(self, x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)

        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return  {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }


    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out
    
    def map2latent(self,x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        x_encoder = x_encoder.permute(0,2,1)
        return x_encoder

    def latent2origin(self,x):
        x = x.permute(0,2,1)
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x, sample_codebook_temp=0.5)
        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity



from vector_quantize_pytorch import ResidualSimVQ

class sim_RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        # rvqvae_config = {
        #     'num_quantizers': args.num_quantizers,
        #     'shared_codebook': args.shared_codebook,
        #     'quantize_dropout_prob': args.quantize_dropout_prob,
        #     'quantize_dropout_cutoff_index': 0,
        #     'nb_code': nb_code,
        #     'code_dim':code_dim, 
        #     'args': args,
        # }
        # self.quantizer = ResidualVQ(**rvqvae_config)


        self.quantizer = ResidualSimVQ(
            dim = 512,
            num_quantizers = 1,
            codebook_size = 512,
            rotation_trick = True  # use rotation trick from Fifty et al.
        )


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def forward(self, x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        x_encoder = x_encoder.permute(0,2,1).contiguous()
        x_quantized, code_idx, commit_loss = self.quantizer(x_encoder)
        commit_loss = commit_loss.mean()
        x_quantized = x_quantized.permute(0,2,1).contiguous()
        perplexity = torch.tensor(0.0).float().cuda()
        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return  {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }


    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out
    
    def map2latent(self,x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        x_encoder = x_encoder.permute(0,2,1)
        return x_encoder

    def latent2origin(self,x):
        x = x.permute(0,2,1)
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x, sample_codebook_temp=0.5)
        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity




def compute_perplexity(nb_code, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

from vector_quantize_pytorch import SimVQ



from torch.nn.utils import weight_norm



class Chomp1d(nn.Module):
    """
    Remove padding to ensure the output length matches the input length
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    Single TCN block with two dilated causal convolutions and residual connection
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Add residual connection if input and output dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        # self.init_weights()

    def init_weights(self):
        # Initialize weights using Xavier initialization
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    TCN network with multiple temporal blocks
    """
    def __init__(self, num_inputs, num_outputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 1#2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Calculate padding to maintain temporal dimension while ensuring causality
            # For causal convolutions, we only pad on the left (past) side
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, 
                                        stride=1, dilation=dilation_size,
                                        padding=padding, dropout=dropout))

        self.network = nn.Sequential(*layers)
        # Final projection layer to output dimension
        self.output_projection = nn.Conv1d(num_channels[-1], num_outputs, 1)

    def forward(self, x):
        # Input shape: [batch, features, time]
        output = self.network(x)
        output = self.output_projection(output)  # Project to the desired output dimension
        return output

class TCN(nn.Module):
    """
    Complete TCN model with 6 layers
    Input dimension: 768
    Output dimension: 106
    Strictly causal implementation
    """
    def __init__(self, input_dim=768, output_dim=106, num_layers=6, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        
        # Define channel sizes for each layer
        # Starting from input_dim and gradually decreasing to prepare for output_dim
        num_channels = [512, 512, 256, 256, 128, 128]
        assert len(num_channels) == num_layers, "Number of channels must match number of layers"
        
        self.tcn = TemporalConvNet(input_dim, output_dim, num_channels, kernel_size, dropout)
        
    def forward(self, x):
        """
        x should have shape (batch_size, input_dim, sequence_length)
        """
        # Apply the TCN layers
        x = x.transpose(1, 2) # (batch_size, sequence_length, input_dim)
        output = self.tcn(x)
        output = output.transpose(1, 2)  # (batch_size, output_dim, sequence_length)
        return output




import argparse
def get_rvqvae_model(ckpt_path = './utils/RVQVAE_6_ckpt/net_300000.pth'):
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    
    args.num_quantizers = 1
    args.shared_codebook =  False
    args.quantize_dropout_prob = 0.2
    args.mu = 0.99

    args.nb_code = 512
    args.code_dim = 512
    args.code_dim = 512
    args.down_t = 2
    args.stride_t = 2
    args.width = 512
    args.depth = 3
    args.dilation_growth_rate = 3
    args.vq_act = "relu"
    args.vq_norm = None
    dim_pose = 315
    model = RVQVAE(args,
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
    re = model.load_state_dict(torch.load(ckpt_path)['net'],strict=False)
    print(f"load model from {ckpt_path}, load result: {re}")
    model.cuda().eval()
    return model



class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)