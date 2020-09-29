import torch
import torch.nn as nn
import torch.nn.Function as F

from torchvision import models

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        net = models.resnet50(pretrained=True)
        
        for param in net.parameters():
            param.requires_grad_(False)

        modules = list(net.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)

        return features


class CaptionNet(nn.Module):
    def __init__(
        self, 
        n_tokens: int, 
        emb_size: int = 128, 
        lstm_units: int = 256, 
        cnn_feature_size: int = 2048
    ):
        super(CaptionNet, self).__init__()


    def forward(self, image_vectors, captions):
        pass




