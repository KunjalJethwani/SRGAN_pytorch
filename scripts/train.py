import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import warnings
warnings.filterwarnings("ignore")

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize, Resize
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from data.dataset import ImageDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image

import warnings
warnings.filterwarnings("ignore")

from models.generator import Generator
from models.discriminator import Discriminator
from models.loss import ContentLoss, PerceptualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_dir = "/content/esrgan_logs"
output_dir = "/content/esrgan_outputs"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Initialize Generator, Discriminator, and Optimizers
generator = Generator()
discriminator = Discriminator()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# Load pre-trained VGG model for Perceptual Loss
vgg = models.vgg19(pretrained=True).to(device)
criterion_content = ContentLoss()
criterion_perceptual = PerceptualLoss(vgg)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Create Dataset and DataLoader
dataset = ImageDataset(root_dir="/data", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


def train(generator, discriminator, dataloader, num_epochs, optimizer_G, optimizer_D, criterion_content, criterion_perceptual, device):
    generator.to(device)
    discriminator.to(device)

    for epoch in range(num_epochs):
        for i, imgs in enumerate(dataloader):
            hr_img = imgs[0].to(device)
            lr_img = imgs[1].to(device)

            # Generate super-resolved image
            sr_image = generator(lr_img)

            # Train Generator
            optimizer_G.zero_grad()
            content_loss = criterion_content(sr_image, hr_img)
            perceptual_loss = criterion_perceptual(sr_image, hr_img)
            g_loss = 0.005*content_loss + perceptual_loss
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_output = discriminator(hr_img)
            fake_output = discriminator(sr_image.detach())
            d_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output)) + \
                     F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))
            d_loss.backward()
            optimizer_D.step()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {i}, G Loss: {g_loss.item()}, D Loss: {d_loss.item()}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "g_scheduler_state_dict": g_scheduler.state_dict(),
            "d_scheduler_state_dict": d_scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(output_dir, "esrgan_checkpoint.pth"))


        # Save generated images
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(output_dir, f"sr_epoch_{epoch+1}.png")
            save_image(sr_image, save_path)

    # Save final models
    torch.save(generator.state_dict(), os.path.join(output_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(output_dir, "discriminator.pth"))

# Train ESRGAN
train(generator, discriminator, dataloader, num_epochs=25, optimizer_G=optimizer_G, optimizer_D=optimizer_D,
      criterion_content=criterion_content, criterion_perceptual=criterion_perceptual, device=device)