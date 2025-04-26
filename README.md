# SRGAN: Super-Resolution Generative Adversarial Network

This repository contains a PyTorch implementation of the **Super-Resolution Generative Adversarial Network (SRGAN)**, as described in the paper ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802). SRGAN is a deep learning model designed to enhance the resolution of low-resolution images and generate visually appealing high-resolution outputs by leveraging adversarial training and perceptual loss.

## Features

- **Generator Network:** Based on a ResNet-like architecture with residual blocks.
- **Discriminator Network:** Uses a convolutional neural network to distinguish between real high-resolution images and generated images.
- **Content Loss:** Perceptual loss computed using features extracted from a pretrained VGG19 network.
- **Adversarial Loss:** Encourages the generator to produce outputs indistinguishable from real high-resolution images.
- **High-Quality Outputs:** The model produces photo-realistic high-resolution images from low-resolution inputs.

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Testing and Visualization](#testing-and-visualization)
5. [Usage](#usage)
6. [Results](#results)
7. [Acknowledgements](#acknowledgements)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10.0 or higher
- CUDA (optional for GPU acceleration)
- Additional libraries: torchvision, matplotlib, Pillow

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/srgan-pytorch.git
    cd srgan-pytorch
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Dataset Preparation

This implementation uses the **DIV2K dataset** for training and testing.

1. Download the dataset from the [official DIV2K website](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
2. Organize the dataset as follows:
    ```
    data/
    |-- DIV2K_train_HR/  # High-resolution images (ground truth)
    |-- DIV2K_train_LR/  # Low-resolution images (optional for validation)
    ```
3. Update the `root_dir` in the `DIV2KDataset` class to point to the dataset directory.

---

## Training

Run the training script:

```bash
python train.py
```

### Key Parameters:
- `batch_size`: Number of images per training batch (default: 16).
- `num_epochs`: Total number of training epochs (default: 100).
- `lr`: Learning rate for both generator and discriminator optimizers (default: 1e-4).
- `scale_factor`: Upscaling factor for SR images (default: 4).
- `crop_size`: Size of random HR crops during training (default: 96).

---

## Testing and Visualization

To test the trained model and visualize results:

1. Use the `test.py` script:
    ```bash
    python test.py --checkpoint generator.pth --input_path ./data/test_images --output_path ./output
    ```

2. Visualize results using Matplotlib:
    ```python
    import matplotlib.pyplot as plt
    from PIL import Image

    sr_image = Image.open("./output/sr_image.png")
    plt.imshow(sr_image)
    plt.title("Super-Resolved Image")
    plt.axis("off")
    plt.show()
    ```

---

## Usage

### Training Custom Dataset

To use a custom dataset, update the `DIV2KDataset` class to handle your data format. Ensure the dataset includes high-resolution images and modify the directory paths accordingly.

### Model Checkpoints

- Save checkpoints after training:
    ```bash
    torch.save(generator.state_dict(), "./output/generator.pth")
    torch.save(discriminator.state_dict(), "./output/discriminator.pth")
    ```
- Load a pretrained generator for inference:
    ```python
    generator.load_state_dict(torch.load("./output/generator.pth"))
    generator.eval()
    ```

---

## Results

Below are example outputs from the SRGAN model:

| Low-Resolution (Input) | High-Resolution (Generated) | Ground Truth |
|-------------------------|----------------------------|--------------|
| ![LR](./examples/lr_image.png) | ![SR](./examples/sr_image.png) | ![HR](./examples/hr_image.png) |

Metrics (calculated on the test set):
- Peak Signal-to-Noise Ratio (PSNR): XX.XX dB
- Structural Similarity Index Measure (SSIM): 0.XXXX

---

## Acknowledgements

This implementation is based on the SRGAN architecture described in the following paper:

- Christian Ledig, Lucas Theis, Ferenc Huszár, et al. *"Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"* ([arXiv:1609.04802](https://arxiv.org/abs/1609.04802))

If you use this repository, please consider citing the paper.

---

Feel free to open issues or contribute improvements to this implementation!

