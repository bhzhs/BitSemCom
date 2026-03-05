import torch.optim as optim
from network.network import WITT, QPSK_soft
from data.datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark = True
import torch
import lpips
from PIL import Image
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils as utils
import traceback
from thop import profile
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

print("All imports successful!")

parser = argparse.ArgumentParser(description='WITT')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--pass_channel', action='store_true',
                    help='pass channel or not')
parser.add_argument('--param', action='store_true',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='DIV2K',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='kodak',
                    choices=['kodak', 'CLIC21','DIV2K','DIV2K_fix','NEW_TEST'],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--pretrain', type=str, default='',
                    help='WITT model or WITT without channel ModNet')
parser.add_argument('--model', type=str, default='WITT',
                    choices=['WITT', 'WITT_W/O'],
                    help='WITT model or WITT without channel ModNet')
parser.add_argument('--channel-type', type=str, default='awgn',
                    choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=int, default=96,
                    help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='10',
                    help='random or fixed snr')
parser.add_argument('--gpu', type=str, default='2',
                    help='# of gpu')
parser.add_argument('--quantize', type=int, default=0,
                    help='quantize level')
parser.add_argument('--px', type=str, default='',
                    help='prex of writter')
parser.add_argument('--save_img', action='store_true',
                    help='save image or not')
parser.add_argument('--bit_error', type=float,default=0.0,
                    help='')
parser.add_argument('--plr', type=float,default=0.0,
                    help='')
parser.add_argument('--segment', type=int,default=0,
                    help='')
parser.add_argument('--snr_min', type=float,default=-3.0,
                    help='')
parser.add_argument('--snr_max', type=float,default=6.0,
                    help='')
parser.add_argument('--test_snr', type=float,default=24.0,
                    help='')
parser.add_argument('--step',type=int,default=50)
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--min_lr',type=float,default=2e-5)
parser.add_argument('--decay',type=float,default=0.9)
parser.add_argument('--tau',type=float,default=1.0)
parser.add_argument('--tau_lg',type=float,default=1.0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
def normalize_for_lpips(x):
    return x * 2 - 1  # [0,1] -> [-1,1]

def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained
def bit_packet_channel(bits, plr):
    if plr > 0:
        bs, bit_len, C = bits.shape
        # 丢包掩码（1=保留，0=丢弃）
        keep_mask = (torch.rand(bs, 1, C, device=bits.device) > plr).float()
        random_bits = torch.randint(0, 2, (bs, 256, C), device=bits.device).float()
        bits = bits * keep_mask + random_bits * (1.0 - keep_mask)
    return bits

def bit_pixel_packet_channel(bits, plr):
    if plr > 0:
        bs, bit_len, C = bits.shape
        # 丢包掩码（1=保留，0=丢弃）
        keep_mask = (torch.rand(bs, bit_len, 1, device=bits.device) > plr).float()
        random_bits = torch.randint(0, 2, (bs, 256, C), device=bits.device).float()
        bits = bits * keep_mask + random_bits * (1.0 - keep_mask)
    return bits

def random_bit_flip(tensor, flip_prob=0.1):
    """
    随机对 tensor 进行比特反转（0 ↔ 1），翻转概率由 flip_prob 控制。

    参数：
    tensor (torch.Tensor): 输入的二值 tensor（元素为 0 或 1）。
    flip_prob (float): 每个元素翻转的概率，取值范围 (0, 1)。

    返回：
    torch.Tensor: 经过随机翻转后的 tensor。
    """
    rand_matrix = torch.rand_like(tensor, dtype=torch.float)  # 生成与 tensor 形状相同的随机概率矩阵
    
    flip_mask = (rand_matrix < flip_prob).to(torch.int8)  # 转换为 int8 掩码
    flipped_tensor = (tensor.to(torch.int8) ^ flip_mask).to(tensor.dtype)  
    return flipped_tensor

def tensor_to_bitstream(tensor, bit_width):
    if bit_width < 1 or bit_width > 32:
        raise ValueError("Bit width should be between 1 and 32")
    
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    range_val = max_val - min_val
    n_levels = 2 ** bit_width
    step_size = range_val / (n_levels - 1)
    
    quantized_tensor = torch.round((tensor - min_val) / step_size)
    quantized_tensor = torch.clamp(quantized_tensor, 0, n_levels - 1)  # Ensure values stay within valid range
    
    # Convert to bitstream (binary string)
    bitstream = []
    for q in quantized_tensor.view(-1):
        # Convert each quantized value to binary with a fixed number of bits
        bitstring = format(int(q.item()), f'0{bit_width}b')  # Fixed-length binary representation
        bitstream.append(bitstring)
    
    return ''.join(bitstream), min_val, step_size, n_levels, tensor.shape

def bitstream_to_tensor(bitstream, bit_width, min_val, step_size, n_levels,original_shape):
    # Convert the bitstream back to quantized values
    num_values = len(bitstream) // bit_width
    quantized_tensor = []
    
    for i in range(num_values):
        bitstring = bitstream[i * bit_width: (i + 1) * bit_width]
        quantized_value = int(bitstring, 2)
        quantized_tensor.append(quantized_value)
    
    quantized_tensor = torch.tensor(quantized_tensor, dtype=torch.float32).view(original_shape).to(config.device)
    
    # Dequantize to get the tensor values
    dequantized_tensor = quantized_tensor * step_size + min_val
    return dequantized_tensor

def introduce_bit_errors(bitstream, ber):
    bitstream = list(bitstream)
    num_bits = len(bitstream)
    num_errors = int(ber * num_bits)
    
    error_indices = np.random.choice(num_bits, num_errors, replace=False)
    
    # Flip the bits at the error indices
    for idx in error_indices:
        bitstream[idx] = '1' if bitstream[idx] == '0' else '0'
    
    return ''.join(bitstream)


def introduce_bit_errors_segments(bitstream, ber, num_segments):
    bitstream = list(bitstream)
    num_bits = len(bitstream)
    
    # 计算每段的长度，并对比特流进行分段
    segment_length = num_bits // num_segments
    segments = [bitstream[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]
    
    # 如果不能整除，将剩余比特附加到最后一段
    if num_bits % num_segments != 0:
        segments[-1].extend(bitstream[num_segments * segment_length:])
    
    # 随机选择一个段
    selected_segment_index = np.random.choice(num_segments)
    selected_segment = segments[selected_segment_index]
    
    # 在选定的段内引入误码
    num_errors = int(ber * len(selected_segment))
    error_indices = np.random.choice(len(selected_segment), num_errors, replace=False)
    
    # 翻转选定段的比特
    for idx in error_indices:
        selected_segment[idx] = '1' if selected_segment[idx] == '0' else '0'
    
    # 将各段重新组合成完整的比特流
    segments[selected_segment_index] = selected_segment
    bitstream = ''.join(''.join(segment) for segment in segments)
    
    return bitstream


class config():
    seed = 1024
    pass_channel = args.pass_channel
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    # logger
    print_step = 1
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]
    if args.training == True:
        workdir = './history/BitSC_Prob/{}'.format(filename)
        log = workdir + '/Log_{}.log'.format(filename)
        samples = workdir + '/samples'
        models = workdir + '/models'
    else:
        workdir = './history/BitSC_Prob_test/{}'.format(filename)
        log = workdir + '/Log_{}.log'.format(filename)
        samples = workdir + '/samples'
        models = workdir + '/models'
    logger = None
    snr_min = args.snr_min
    snr_max = args.snr_max
    # training details
    normalize = False
    learning_rate = args.lr
    tot_epoch = 10000000

    # Add training configs
    min_lr = 1e-6  # Minimum learning rate
    grad_clip = 1.0  # Gradient clipping threshold
    patience = 5  # Patience for learning rate reduction
    lr_factor = 0.5  # Factor to reduce learning rate
    early_stop_patience = 15  # Early stopping patience

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        train_data_dir = "/media/Dataset/CIFAR10/"
        test_data_dir = "/media/Dataset/CIFAR10/"
        batch_size = 128
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 20
        image_dims = (3, 256, 256)
        train_data_dir = ["/data/zhs/WITT/DIV2K_train_HR/DIV2K_train_HR/"]
        # train_data_dir=['/home/zsh/etf/openimage/open_image_train']
        if args.testset == 'kodak':
            test_data_dir = ["kodak_patches_all"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["/media/Dataset/CLIC21/"]
        elif args.testset == 'DIV2K':
            test_data_dir = ["/data/zhs/WITT/media/DIV2K_valid_HR/DIV2K_valid_HR"]
        
        elif args.testset == 'DIV2K_fix':
            test_data_dir = ["DIV2K"]
        
        elif args.testset == 'NEW_TEST':
            test_data_dir = ["NEW_TEST"]

        batch_size = 16
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    
    # 确保工作目录存在
    os.makedirs(workdir, exist_ok=True)

if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained

def load_weights_false(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=False)
    del pretrained


def train_one_epoch(args,batch_snr):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    
    global global_step
    total_loss = 0
    batch_count = 0
    results = np.zeros((len(train_loader), 4))
    for batch_idx, data in enumerate(train_loader):
        start_time = time.time()
        global_step += 1
        input = data.cuda() if isinstance(data, torch.Tensor) else data[0].cuda()
        recon_image, CBR, SNR, mse, loss_G = net(input,batch_snr)
        loss = loss_G
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新指标
        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        cbrs.update(CBR)
        snrs.update(SNR)
        
        if mse.item() > 0:
            psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
            psnrs.update(psnr.item())
            msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
            msssims.update(msssim)
        else:
            psnrs.update(100)
            msssims.update(100)
        
        total_loss += loss.item()
        batch_count += 1
        
        if (global_step % config.print_step) == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            print(' | '.join([
                f'Epoch {epoch}',
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Time {elapsed.val:.3f}',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                f'Lr {cur_lr}',
            ]))
        results[batch_idx-1] =[mse.item(), CBR, psnr.item(), msssim]
        # 计算平均损失
    cbr_,psnr_,ssim_ = results[:,1].mean(),results[:,2].mean(),results[:,3].mean()
    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    
    writer.add_scalar('Train/PSNR', psnr_, epoch)
    writer.add_scalar('Train/SSIM', ssim_, epoch)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

    print(f"Epoch completed with average loss: {avg_loss}")

    return avg_loss
        

def test(batch_snr):
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            if args.trainset == 'CIFAR10':
                for batch_idx, (input, label) in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, batch_snr)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    print(' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
            else:
                for batch_idx, input in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, batch_snr)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    print(' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()

    writer.add_scalar('Test/PSNR',results_psnr[0], epoch)
    writer.add_scalar('Test/SSIM', results_msssim[0], epoch)

    print("SNR: {}" .format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}" .format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")



def test_woruns():
    config.isTrain = False
    net.eval()
    enc_times,dec_times, psnrs, msssims, snrs, cbrs,lpipss = [AverageMeter() for _ in range(7)]
    metrics = [enc_times,dec_times, psnrs, msssims, snrs, cbrs,lpipss]
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()  # 或 'vgg', 'squeeze'
    
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_enct = np.zeros(len(multiple_snr))
    results_dect = np.zeros(len(multiple_snr))
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    results_lpips = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            for batch_idx, input in enumerate(test_loader):
                # start_time = time.time()
                input = input.cuda()
                if args.param:
                    flops_encoder, params_encoder = profile(net.encoder, inputs=(input,SNR,net.model,))
                enc_begin = time.time()
                
                logits, y_prob, feature = net.Encoder(input, args.test_snr)
                
                    
                enc_time = time.time() - enc_begin
                
                received, received_Prob = QPSK_soft(feature, args.test_snr)
                
                if batch_idx == 2:
                    output_dir = "./test_probs_C%s/tau_%s" % (args.C,args.tau)
                    os.makedirs(output_dir, exist_ok=True)  # 如果目录已存在，不会报错
                    
                    import matplotlib.pyplot as plt
                    logits_flat = logits[...,1].detach().cpu().numpy().reshape(-1)
                    y_prob_flat = y_prob[...,1].detach().cpu().numpy().reshape(-1)
                    rx_prob_flat = received_Prob.detach().cpu().numpy().reshape(-1)
                    np.save(os.path.join(output_dir, "logits_flat_%sdB.npy" % args.test_snr), logits_flat)
                    np.save(os.path.join(output_dir, "y_prob_flat_%sdB.npy" % args.test_snr), y_prob_flat)
                    np.save(os.path.join(output_dir, "rx_prob_flat_%sdB.npy" % args.test_snr), rx_prob_flat)

            
                    
                    plt.figure(figsize=(6, 4))
                    plt.hist(logits_flat, bins=200)
                    # plt.scatter(logits_flat)
                    plt.xlabel("Logit Value")
                    plt.ylabel("Frequency")
                    plt.title("Distribution of Logits")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "logits_%sdB.png" % args.test_snr), dpi=300)
                    
                    plt.figure(figsize=(6, 4))
                    plt.hist(y_prob_flat, bins=200)
                    # plt.scatter(y_prob_flat)
                    plt.xlabel("Prob Value")
                    plt.ylabel("Frequency")
                    plt.title("Distribution of Prob")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "Probs_%sdB.png" % args.test_snr), dpi=300)
                    
                    plt.figure(figsize=(6, 4))
                    plt.hist(rx_prob_flat, bins=200)
                    # plt.scatter(y_prob_flat)
                    plt.xlabel("Rx Prob Value")
                    plt.ylabel("Frequency")
                    plt.title("Distribution of Received Prob")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "Rx_Probs_%sdB.png" % args.test_snr), dpi=300)
                    
                    

                # bitstream, min_val, step_size, n_levels,tensor_shape = tensor_to_bitstream(feature, args.quantize)
                # print(tensor_shape)
                # if args.segment == 0:
                #     bitstream_with_errors = introduce_bit_errors(bitstream, args.bit_error)
                # else:
                #     bitstream_with_errors = introduce_bit_errors_segments(bitstream,args.bit_error,args.segment)

                # feature = bitstream_to_tensor(bitstream_with_errors, args.quantize, min_val, step_size, n_levels,tensor_shape)

                # feature = random_bit_flip(feature, flip_prob=args.bit_error)

                CPR = feature.numel()/(256*256*3*8)
                
                if args.param:
                    flops_decoder, params_decoder = profile(net.decoder, inputs=(received_Prob,args.test_snr,net.model,))
                
                dec_begin = time.time()
                recon_image, CBR, SNR, mse, loss_G = net.Decoder(input, received_Prob, args.test_snr)
                dec_time = time.time()-dec_begin

                enc_times.update(enc_time)
                dec_times.update(dec_time)
                cbrs.update(CBR)
                snrs.update(SNR)
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                else:
                    psnrs.update(100)
                    msssims.update(100)
                lpipss_d = loss_fn_alex(normalize_for_lpips(recon_image), normalize_for_lpips(input)).mean().item()
                lpipss.update(lpipss_d)
                print(' | '.join([
                    f'Enc_time {enc_times.val:.3f}',
                    f'Dec_time {dec_times.val:.3f}',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f}',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))

                if args.save_img:
                    if args.bit_error==0:
                        output_folder = './Image_samples/' + args.testset+'_output'
                    else:
                        output_folder = args.testset+'_output_%s_seg%s'%(args.bit_error,args.segment)
                    os.makedirs(output_folder, exist_ok=True) 
                    recon_image_np = recon_image.squeeze().cpu().clamp(0, 1).numpy()  # 转换为 NumPy 数组
                    recon_image_np = (recon_image_np * 255).astype(np.uint8).transpose(1, 2, 0)
                    print(recon_image_np.shape)
                    if recon_image_np.shape[0] == 1:
                        recon_image_np = recon_image_np[0]
                    
                    recon_img = Image.fromarray(recon_image_np)
                    recon_img.save(os.path.join(output_folder, f"recon_img_{batch_idx}_{CPR:.4f}_p{psnr.item():3f}_s{msssim:.3f}.png"))
        results_enct[i] = enc_times.avg
        results_dect[i] = dec_times.avg
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        results_lpips[i] = lpipss.avg
        for t in metrics:
            t.clear()
    
    print("CPR: {}" .format(CPR))
    print("Enc time: {}" .format(results_enct.tolist()))
    print("Dec time: {}" .format(results_dect.tolist()))
    print("SNR: {}" .format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}" .format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("LPIPS: {}".format(results_lpips.tolist()))
    if args.param:
        print(f"Encoder FLOPs: {flops_encoder / 1e9:.2f} GFLOPs")
        print(f"Encoder 参数量: {params_encoder / 1e6:.2f} M")

        print(f"Decoder FLOPs: {flops_decoder / 1e9:.2f} GFLOPs")
        print(f"Decoder 参数量: {params_decoder / 1e6:.2f} M")

    print("Finish Test!")




if __name__ == '__main__':

    # 初始化阶段
    print("Initializing random seed...")
    seed_torch()
    
    print("Setting up logger...")
    logger = logger_configuration(config, save_log=True)
    print("Config details:")
    print(config.__dict__)
    
    print("Setting up model...")
    torch.manual_seed(seed=config.seed)
    net = WITT(args, config)
    net = net.cuda()
    print("Model moved to CUDA")
    if args.pretrain!='':
        load_weights_false(args.pretrain)
        print('*' * 50)
        print('load model parameters ...')


    
    print("Setting up optimizer...")
    model_params = [{'params': net.parameters(), 'lr': args.lr}]
    
    print("Loading datasets...")
    train_loader, test_loader = get_loader(args, config)

    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    print(f"Steps per epoch: {steps_epoch}")

    # 设置学习率调度器
    print("Setting up learning rate scheduler...")
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor, 
    #                             patience=config.patience, min_lr=config.min_lr)
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.decay)
    # 设置早停
    best_loss = float('inf')
    no_improve = 0
    min_lr = args.min_lr
    if args.training:
        if args.pretrain=='':
            if args.pass_channel== False:
                writer = SummaryWriter(log_dir='./runs_tb_kodak/BitSC_Prob/'+'%s_C%s_lr%s_snr%s_%s/'%(args.px,args.C,args.lr,args.snr_min,args.snr_max))
            if args.pass_channel == True:
                writer = SummaryWriter(log_dir='./runs_tb_kodak/BitSC_Prob/'+'%s_C%s_lr%s_pc_snr%s_%s/'%(args.px,args.C,args.lr,args.snr_min,args.snr_max))
        elif args.pretrain!='':
            if args.pass_channel== False:
                writer = SummaryWriter(log_dir='./runs_tb_kodak/BitSC_Prob/'+'%s_Pre_C%s_lr%s_snr%s_%s/'%(args.px,args.C,args.lr,args.snr_min,args.snr_max))
            if args.pass_channel == True:
                writer = SummaryWriter(log_dir='./runs_tb_kodak/BitSC_Prob/'+'%s_Pre_C%s_lr%s_pc_snr%s_%s/'%(args.px,args.C,args.lr,args.snr_min,args.snr_max))

        print("=== Starting training loop ===")
        for epoch in range(steps_epoch, config.tot_epoch):
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
            batch_snr = random.randint(args.snr_min, args.snr_max)
            try:
                # 训练一个epoch
                train_loss = train_one_epoch(args,batch_snr)
                # 学习率调度
                scheduler.step()
                for param_group in optimizer.param_groups:
                    cur_lr = param_group['lr']
                    if param_group['lr'] < min_lr:
                        param_group['lr'] = min_lr

                # 定期保存和测试
                if (epoch + 1) % config.save_model_freq == 0:
                    print(f"=== Running periodic test at epoch {epoch + 1} ===")
                    save_model(net, save_path=config.models + '/{}_EP{}.model'
                                .format(config.filename, epoch + 1))
                    test(batch_snr)
                
            except Exception as e:
                print(f"Error during epoch {epoch}: {str(e)}")
                print(f"Stack trace: {traceback.format_exc()}")
                raise e
            
    else:
        epoch = 0
        print("=== Starting test mode ===")
        test_woruns()

            


