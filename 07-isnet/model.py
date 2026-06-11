"""
Model definition script for the IS-Net segmentation experiment.

Defines the SimpleSegModel class, a lightweight Convolutional Neural Network (CNN)
structured as an encoder-decoder architecture. Used to predict binary 
segmentation probability masks from 3-channel (RGB) input images.
"""

import torch
import torch.nn as nn

class SimpleSegModel(nn.Module):
    """
    A lightweight Convolutional Neural Network for binary image segmentation.
    
    Architecture:
        - Encoder: 2 downsampling blocks (Conv2d -> ReLU -> MaxPool2d)
          to extract low and mid-level feature representations.
        - Decoder: 2 upsampling blocks (ConvTranspose2d -> ReLU) 
          and a final convolution with Sigmoid activation to produce
          a binary segmentation mask (values between 0 and 1).
    """
    def __init__(self):
        super().__init__()

        # Encoder extracts features and reduces spatial dimensions: (3, 256, 256) -> (32, 64, 64)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (16, 128, 128)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # (32, 64, 64)
        )

        # Decoder upsamples the feature maps back to original input resolution: (32, 64, 64) -> (1, 256, 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # (16, 128, 128)
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),   # (1, 256, 256)
            nn.Sigmoid()  # Restrict outputs to [0, 1] range representing probability of foreground
        )

    def forward(self, x):
        """
        Forward pass for the model.
        
        Parameters:
            x (torch.Tensor): Input image batch of shape (B, 3, H, W).
            
        Returns:
            torch.Tensor: Predicted segmentation mask of shape (B, 1, H, W).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
