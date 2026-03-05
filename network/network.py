# 导入必要的模块
from network.decoder import *
from network.encoder import *
from loss.distortion import Distortion
from network.channel import Channel

from random import choice
import torch.nn as nn
def add_quantization_noise(tensor, quantization_levels=255, noise_type='uniform'):
    """
    对输入张量进行 8 比特量化并加入噪声。

    参数:
    - tensor (torch.Tensor): 输入张量，假设值在 [0, 1] 范围内。
    - quantization_levels (int): 量化级别，默认 8 比特量化为 255。
    - noise_type (str): 噪声类型，'uniform' 表示均匀分布，'normal' 表示正态分布。
    
    返回:
    - torch.Tensor: 加入量化噪声后的张量。
    """
    # 1. 将张量缩放到 [0, quantization_levels] 范围
    tensor_scaled = tensor * quantization_levels

    # 2. 根据噪声类型添加噪声
    if noise_type == 'uniform':
        noise = torch.rand_like(tensor_scaled) - 0.5  # 均匀分布噪声 [-0.5, 0.5)
    elif noise_type == 'normal':
        noise = torch.randn_like(tensor_scaled) * 0.5  # 标准差为 0.5 的正态分布噪声
    else:
        raise ValueError("Unsupported noise type. Choose 'uniform' or 'normal'.")

    tensor_noisy_scaled = tensor_scaled + noise

    # 3. 四舍五入并裁剪到 [0, quantization_levels] 范围
    tensor_quantized = torch.clamp(torch.round(tensor_noisy_scaled), 0, quantization_levels)

    # 4. 将张量恢复到原始范围 [0, 1]
    tensor_noisy = tensor_quantized / quantization_levels

    return tensor_noisy

def bit_channel(bits, ber):
    if ber > 0:
        p_i = torch.rand_like(bits)  
        flip_mask = (p_i < ber).float()  
        bits = bits * (1 - flip_mask) + (1 - bits) * flip_mask  
    return bits

def BPSK_soft(bit_tensor, SNR):
    bpsk_symbols = 1.0 - 2.0 * bit_tensor.float()  # 0 -> +1, 1 -> -1
    snr_linear = 10 ** (SNR / 10)
    snr_linear_tensor = torch.tensor(snr_linear, dtype=torch.float32, device=bit_tensor.device)
    noise_std = torch.sqrt(1 / (2 * snr_linear_tensor))  # AWGN: σ = sqrt(N0/2)
    awgn_noise = noise_std * torch.randn_like(bpsk_symbols)
    received = bpsk_symbols + awgn_noise
    llr = 2 * received / (noise_std ** 2)
    return received, llr



def QPSK_soft(bit_tensor, SNR):
    """
    Performs QPSK modulation, adds AWGN, and computes LLRs using real-valued tensors.
    This version correctly handles multi-dimensional input tensors (e.g., [batch, channels, features]).
    
    Args:
        bit_tensor (torch.Tensor): A tensor of bits (0s and 1s). 
                                   The size of its LAST dimension must be even.
        SNR (float): The signal-to-noise ratio in dB, interpreted as Es/N0.

    Returns:
        torch.Tensor: The received symbols, with an added dimension of size 2 for I/Q.
                      Shape: [..., W/2, 2]
        torch.Tensor: The calculated LLR for each bit, with the same shape as bit_tensor.
    """
    # 1. Input Validation and Reshaping
    if bit_tensor.shape[-1] % 2 != 0:
        raise ValueError("The last dimension of the bit_tensor must be even for QPSK.")
    
    # Correctly reshape to preserve batch and other dimensions
    prefix_shape = bit_tensor.shape[:-1]
    last_dim = bit_tensor.shape[-1]
    bit_pairs = bit_tensor.view(*prefix_shape, last_dim // 2, 2)

    # 2. Bit-to-Symbol Mapping (Gray Coding)
    # --- FIX 1: Use ellipsis (...) for correct multi-dimensional slicing ---
    symbols_I = 1.0 - 2.0 * bit_pairs[..., 0].float()
    symbols_Q = 1.0 - 2.0 * bit_pairs[..., 1].float()
    
    # 3. Constellation Creation
    scaling_factor = torch.sqrt(torch.tensor(2.0, device=bit_tensor.device))
    qpsk_symbols = torch.stack([symbols_I, symbols_Q], dim=-1) / scaling_factor

    # 4. Noise Calculation
    snr_linear = 10 ** (SNR / 10.0)
    snr_linear_tensor = torch.tensor(snr_linear, dtype=torch.float32, device=bit_tensor.device)
    noise_variance = 1 / (2 * snr_linear_tensor)
    noise_std = torch.sqrt(noise_variance)
    
    awgn_noise = noise_std * torch.randn_like(qpsk_symbols)
    
    # 5. Add noise
    received = qpsk_symbols + awgn_noise

    # 6. Soft Demodulation (LLR Calculation)
    # The LLR for a bit in BPSK is LLR = 2*y*A / sigma^2, where A is the amplitude.
    # For our QPSK, the amplitude A on each (I/Q) dimension is 1/sqrt(2).
    # So, LLR = 2 * received_component * (1/sqrt(2)) / noise_variance
    # LLR = sqrt(2) * received_component / noise_variance
    llr_I = scaling_factor * received[..., 0] / noise_variance
    llr_Q = scaling_factor * received[..., 1] / noise_variance

    # 7. Reshape LLRs to match original bit_tensor dimension
    # Stack the LLRs for the I and Q channels and then flatten back
    llrs_stacked = torch.stack([llr_I, llr_Q], dim=-1)
    llrs_reshaped = llrs_stacked.view(*prefix_shape, last_dim)
    
    # 8. Convert LLR to Probability
    # P(bit=1 | y) = sigmoid(-LLR)
    # A large positive LLR means bit 0 is likely -> probability of bit 1 is low.
    # A large negative LLR means bit 1 is likely -> probability of bit 1 is high.
    # The sigmoid function correctly maps this to the [0, 1] range.
    prob_one = torch.sigmoid(-llrs_reshaped)
    return received, prob_one



class Prob_Layer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Prob_Layer, self).__init__()
        # 定义 1D 转置卷积层，输出形状将是 (bs, output_channels, sequence_length)
        self.trans_conv1d = nn.ConvTranspose1d(
            in_channels=input_channels, 
            out_channels=output_channels, 
            kernel_size=3,  # 卷积核大小
            stride=1,       # 步长
            padding=1       # 填充
        )

    def forward(self, x):
        bs, seq_len, c = x.shape  # 输入形状 (bs, 256, C)
        # 调整形状以适配 1D 转置卷积
        x = x.permute(0, 2, 1)  # 转换为 (bs, C, 256)
        x = self.trans_conv1d(x)  # 应用 1D 转置卷积，输出形状为 (bs, 2*C, 256)
        x = x.permute(0, 2, 1)  # 转回 (bs, 256, 2*C)
        x = x.view(bs, seq_len, c, 2)  # 转换为 (bs, 256, C, 2)
        return x

class Resample_Layer(nn.Module):
    def __init__(self, input_channels):
        super(Resample_Layer, self).__init__()
        self.prob_layer = Prob_Layer(input_channels, input_channels * 2)

    def forward(self, x,tau,tau_lg=1.0):
        logits = self.prob_layer(x)/tau_lg  # 获取 logits bs x 256 x C x 2
        discrete_code = F.gumbel_softmax(logits, hard=True,tau=tau, dim=-1)
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(-1)
        
        
        output = discrete_code[:,:,:,0] * 0 + discrete_code[:,:,:,1] * 1
        # print(output[0,0,:10])
        return logits,y_soft,output


class WITT(nn.Module):
    def __init__(self, args, config):
        super(WITT, self).__init__()
        self.config = config
        self.tau = args.tau
        self.tau_lg = args.tau_lg
        # 获取编码器和解码器的配置参数
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        # 创建编码器和解码器实例
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        
        # 记录网络配置信息
        if config.logger is not None:
            config.logger.info("SNR_range:%sdB---%sdB"%(config.snr_min,config.snr_max))
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
            config.logger.info("tau_lg: ")
            config.logger.info(args.tau_lg)
            
        # 初始化损失函数、信道模型等组件
        self.distortion_loss = Distortion(args)  # 失真损失函数
        self.channel = Channel(args, config)     # 信道模型
        self.pass_channel = config.pass_channel  # 是否通过信道
        self.squared_difference = torch.nn.MSELoss(reduction='none')  # MSE损失
        
        # 初始化图像尺寸参数
        self.H = self.W = 0
        
        # 处理SNR参数列表
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.resample = Resample_Layer(args.C)

        self.downsample = config.downsample  # 下采样率
        self.model = args.model              # 模型类型

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        device = feature.device
        logits,y_prob,feature_dis = self.resample(feature,self.tau,self.tau_lg)
        # uniform_noise = (torch.rand_like(feature, device=device) - 0.5) * (1/256) # 8 bit width
        # feature = feature + uniform_noise
        # noisy_feature = self.channel.forward(feature_dis, chan_param, avg_pwr)
        return logits, y_prob, feature_dis

    def forward(self, input_image,given_SNR = None):
        # 获取输入图像的尺寸
        B, _, H, W = input_image.shape

        # 如果图像尺寸发生变化，更新编码器和解码器的分辨率
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        # 确定SNR参数
        if given_SNR is None:
            # 如果未指定SNR，随机选择一个
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        # 编码器前向传播
        feature = self.encoder(input_image, chan_param, self.model) # shape : 

        # 计算压缩比率(CBR)
        CBR = feature.numel() / 2 / input_image.numel()
        
        # 特征通过信道
        if self.pass_channel:
            logits,y_prob, noisy_feature = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature 
        # print(noisy_feature.shape)
        received, received_llr = QPSK_soft(noisy_feature, given_SNR)
    
        recon_image = self.decoder(received_llr, chan_param, self.model)
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()
    
    
    def Encoder(self,input_image, given_SNR = None):
        B, _, H, W = input_image.shape
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W
        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR
        feature = self.encoder(input_image, chan_param, self.model)
        if self.pass_channel:
            logits, y_prob, noisy_feature = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature
        return logits,y_prob,noisy_feature
    
    def Decoder(self,input_image,feature,given_SNR = None):
        B, _, H, W = input_image.shape
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        # 确定SNR参数
        if given_SNR is None:
            # 如果未指定SNR，随机选择一个
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR
        CBR = feature.numel() / 2 / input_image.numel()

        # feature = add_quantization_noise(feature,quantization_levels=255,noise_type='uniform')

        recon_image = self.decoder(feature, chan_param, self.model)
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()