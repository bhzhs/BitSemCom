import numpy as np
import math
import torch
import random
import os
import logging
import time


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def logger_configuration(config, save_log=False, test_mode=False):
    # 配置 logger
    logger = logging.getLogger("Deep joint source channel coder")
    if test_mode:
        config.workdir += '_test'
    if save_log:
        makedirs(config.workdir)
        makedirs(config.samples)
        makedirs(config.models)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CB_Datasets(Dataset):
    def __init__(self, data_dir):
        super(CB_Datasets, self).__init__()
        self.data_dir = data_dir
        self.imgs = []
        # Support a list of directories
        if isinstance(data_dir, list):
            for dir_path in self.data_dir:
                self.imgs.extend(glob(os.path.join(dir_path, '*.jpg')))
                self.imgs.extend(glob(os.path.join(dir_path, '*.png')))
        else: # Support a single directory string
            self.imgs.extend(glob(os.path.join(data_dir, '*.jpg')))
            self.imgs.extend(glob(os.path.join(data_dir, '*.png')))
        
        self.imgs.sort()

    def __getitem__(self, item):
        image_path = self.imgs[item]
        image = Image.open(image_path).convert('RGB')
        
        # Crop dimensions to be divisible by a factor (e.g., 128)
        # This ensures consistent tensor sizes for some architectures
        w, h = image.size
        new_h = h - h % 128
        new_w = w - w % 128
        
        # Define transform inside __getitem__ to use per-image dimensions
        transform = transforms.Compose([
            transforms.CenterCrop((new_h, new_w)), # Note: PIL uses (h, w) for crop
            transforms.ToTensor()
        ])
        
        img_tensor = transform(image)
        
        # Return both the image tensor and its original path
        return img_tensor, image_path

    def __len__(self):
        return len(self.imgs)
    
import torch
import numpy as np

def save_indices_to_bitstream(indices: torch.Tensor, num_bits_per_index: int, output_path: str):
    """
    Packs a tensor of integer indices into a true bitstream and saves it as a binary file.
    This version correctly handles non-byte-aligned bit widths (like 12-bit).

    Args:
        indices (torch.Tensor): A 1D or 2D tensor of integer indices.
        num_bits_per_index (int): The number of bits for each index (e.g., 12 for a 4096-sized codebook).
        output_path (str): The path to save the resulting binary file.
    """
    if num_bits_per_index <= 0 or num_bits_per_index > 32:
        raise ValueError("num_bits_per_index must be between 1 and 32.")

    # Flatten the indices tensor and move to CPU
    indices_flat = indices.squeeze().cpu().numpy().astype(np.uint32)
    
    # Header: We'll save the number of indices and bit width to help with decoding
    # This makes the file self-contained.
    # [ number_of_indices (4 bytes), num_bits_per_index (1 byte) ]
    header = np.array([len(indices_flat), num_bits_per_index], dtype=np.uint32)
    
    bit_buffer = 0
    bits_in_buffer = 0
    byte_list = bytearray()

    for index in indices_flat:
        # Add the new index's bits to the buffer
        # The `<<` operator is a left bit shift
        bit_buffer = (bit_buffer << num_bits_per_index) | index
        bits_in_buffer += num_bits_per_index
        
        # While there's at least one full byte in the buffer, write it out
        while bits_in_buffer >= 8:
            # Isolate the top 8 bits
            bits_to_write = bit_buffer >> (bits_in_buffer - 8)
            byte_list.append(bits_to_write)
            
            # Remove the bits we just wrote from the buffer
            bits_in_buffer -= 8
            # Use a bitmask to keep only the remaining lower bits
            mask = (1 << bits_in_buffer) - 1
            bit_buffer &= mask
            
    # If there are any remaining bits in the buffer after the loop, pad and write them
    if bits_in_buffer > 0:
        # Left-shift to align the remaining bits to the most significant side of a byte
        bits_to_write = bit_buffer << (8 - bits_in_buffer)
        byte_list.append(bits_to_write)

    # Write the header and the packed data to the file
    with open(output_path, 'wb') as f:
        f.write(header.tobytes())
        f.write(byte_list)


def load_bitstream_to_indices(input_path: str, num_indices,num_bits_per_index):
    """
    Reads a binary bitstream file and correctly decodes it back into a tensor of indices.
    (Corrected Version)

    Args:
        input_path (str): The path to the binary file.

    Returns:
        torch.Tensor: The decoded integer indices.
    """
    with open(input_path, 'rb') as f:
        # Read the header: 4 bytes for num_indices, 4 bytes for num_bits
        header_bytes = f.read(8)
        header = np.frombuffer(header_bytes, dtype=np.uint32)

        byte_list = f.read()

    indices = []
    bit_buffer = 0
    bits_in_buffer = 0
    
    # Create a mask to extract a single index (e.g., for 12 bits, this is 4095)
    index_mask = (1 << num_bits_per_index) - 1

    for byte in byte_list:
        # Shift the existing buffer left by 8 and add the new byte
        bit_buffer = (bit_buffer << 8) | byte
        bits_in_buffer += 8

        # While there are enough bits in the buffer to extract at least one index
        while bits_in_buffer >= num_bits_per_index:
            # Isolate the top `num_bits_per_index` bits to get the index
            shift_amount = bits_in_buffer - num_bits_per_index
            index = (bit_buffer >> shift_amount) & index_mask
            
            # We must not read more indices than are specified in the header
            if len(indices) < num_indices:
                indices.append(index)
            
            # --- THIS IS THE FIX ---
            # Remove the bits we just read from the buffer by applying a mask
            # for the remaining bits.
            bits_in_buffer -= num_bits_per_index
            remaining_mask = (1 << bits_in_buffer) - 1
            bit_buffer &= remaining_mask

    return torch.tensor(indices, dtype=torch.long)