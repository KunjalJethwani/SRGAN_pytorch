import os
import sys
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.generator import Generator
from models.discriminator import Discriminator
from models.loss import VGGFeatureExtractor, SRGANLoss
from data.dataset import DIV2KDataset

# Function to denormalize images (if they were normalized to [-1, 1])
def denormalize(img_tensor):
    return (img_tensor + 1.0) / 2.0  # Scale back to [0, 1]

# Function to calculate PSNR and SSIM
def calculate_metrics(generator, dataloader, device):
    psnr_total = 0
    ssim_total = 0
    count = 0

    generator.eval()  # Set generator to evaluation mode

    with torch.no_grad():
        for lr, hr in dataloader:
            lr = lr.to(device)
            hr = hr.to(device)
            # print(f"LR shape: {lr.shape}")
            # Generate SR images
            sr = generator(lr)

            # Denormalize images
            sr = denormalize(sr).cpu().numpy()
            hr = denormalize(hr).cpu().numpy()
            # print(f"SR shape: {sr.shape}, HR shape: {hr.shape}")


            # Convert tensors to numpy arrays for metric calculations
            sr = np.transpose(sr, (0, 2, 3, 1))  # Convert to (batch, height, width, channels)
            hr = np.transpose(hr, (0, 2, 3, 1))
            # print(f"SR shape: {sr.shape}, HR shape: {hr.shape}")


            for i in range(sr.shape[0]):  # Iterate through the batch
                psnr_total += peak_signal_noise_ratio(hr[i], sr[i], data_range=1.0)
                ssim_total += structural_similarity(hr[i], sr[i], multichannel=True, data_range=1.0, channel_axis=-1)
                count += 1

    # Average metrics over all images
    avg_psnr = psnr_total / count
    avg_ssim = ssim_total / count

    return avg_psnr, avg_ssim

# Load your dataset and trained generator
hr = "data\\DIV2K_valid_HR"
lr = "data\\DIV2K_valid_LR"
scale_factor = 4
crop_size = 96
batch_size = 16  # Set batch size to 16

dataset = DIV2KDataset(hr, lr, scale_factor=scale_factor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Load trained generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = os.path.join("D:\\Clg\\sem6\\NN-2\\SRGAN", "srgan_output\\generator.pth")
generator = Generator().to(device)
generator.load_state_dict(torch.load(gen, map_location=device))

# Calculate PSNR and SSIM
psnr, ssim = calculate_metrics(generator, dataloader, device)

print(f"Average PSNR: {psnr:.2f}")
print(f"Average SSIM: {ssim:.4f}")
