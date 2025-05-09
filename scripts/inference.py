import warnings
warnings.filterwarnings("ignore")
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize, Resize
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import sys
import warnings
warnings.filterwarnings("ignore")

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.generator import Generator
from models.discriminator import Discriminator
from models.loss import VGGFeatureExtractor, SRGANLoss
from data.dataset import DIV2KDataset

# Load a test image
test_image = Image.open("BSDS300/images/test/3096.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

test_image = transform(test_image).unsqueeze(0).to(device)

# Generate super-resolved image
generator.eval()
with torch.no_grad():
    sr_image = generator(test_image)

# Save and Display Results
save_image(sr_image, "sr_image.png")
save_image(test_image, "lr_image.png")

# Show images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Low-Resolution Image")
plt.imshow(np.transpose(test_image.squeeze().cpu().numpy(), (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.title("Super-Resolved Image")
plt.imshow(np.transpose(sr_image.squeeze().cpu().numpy(), (1, 2, 0)))
plt.show()