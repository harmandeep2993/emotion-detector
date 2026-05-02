#models/emotion_model.py

"""Emotion recognition model using a pre-trained ResNet18."""

import torch
import torch.nn as nn
from torchvision import models


# Define the emotion recognition model
class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionResNet, self).__init__()

        # load pretrained ResNet18 — weights trained on ImageNet
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # ResNet18 expects 3-channel (RGB) input
        # our face images are grayscale (1 channel)
        # so we replace the first conv layer to accept 1 channel
        self.model.conv1 = nn.Conv2d(
            in_channels=1,       # grayscale
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # replace the final fully connected layer
        # original ResNet18 outputs 1000 classes (ImageNet)
        # we need 7 classes (7 emotions)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# emotion labels — same order as FER2013 dataset
EMOTIONS = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]
 

def get_model():
    model = EmotionResNet(num_classes=7)
    model.eval()  # set to inference mode — disables dropout/batchnorm training behaviour
    return model
