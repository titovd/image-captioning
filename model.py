import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.cnn_to_h0 = nn.Linear(cnn_feature_size, lstm_units)
        self.cnn_to_c0 = nn.Linear(cnn_feature_size, lstm_units)

        self.emb = nn.Embedding(n_tokens, emb_size)
        self.lstm = nn.LSTM(emb_size, lstm_units, batch_first=True)
        self.logits = nn.Linear(lstm_units, n_tokens)

    def forward(self, image_vectors, captions):
        initial_cell = self.cnn_to_c0(image_vectors)
        initial_hidden = self.cnn_to_hidden(image_vectors)

        captions_emb = self.emb(captions)
        lstm_out, _ = self.lst(captions_emb, (initial_hidden[None], initial_cell[None]))
        logits = self.lstm(lstm_out)

        return logits


