import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    '''A feed-forward neural network encoder with 4 convolutional layers.'''
    def __init__(self):
        super(Encoder, self).__init__()

        #Encoder: compresses the input image to a lower-dimensional representation.
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1), #Dimensions: 512, 288, 8 (channels)
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), #Dimensions: 256, 144, 16
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), #Dimensions: 128, 72, 32
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        )
        
    def forward(self,x):
        '''The forward pass of the encoder.'''
        encoded = self.encoder(x)
        return encoded

class ClassifierHead(nn.Module):
    '''A simple non-linear classifier head with two fully connected layers.'''
    def __init__(self, in_channels=64, hidden_dim=512, num_classes=2):
        super().__init__()

        #Multi-channel feature map is pooled to produce a flat feature vector.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
        #Linear transformation from 32 (in_channels) to 256 (hidden_dim) features.
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        #Linear transformation from 256 (hidden_dim) features to 2 class logit scores.
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        '''The forward pass of the classifier head.'''
        x = self.pool(features).flatten(1) #From [B, C, H, W] to [B, C].

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        logits = self.fc2(x) #[B, 2]
        return logits

class EmbeddingHead(nn.Module):
    '''Small head to convert a feature map into a vector embedding for triplet loss.'''
    def __init__(self, in_channels=64, out_dim=128):
        super().__init__()

        #Multi-channel feature map is pooled to produce a flat feature vector.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        #Linear transformation from 32 (in_channels) to 128 (out_dim) features.
        self.fc = nn.Linear(in_channels, out_dim)

    def forward(self, feat):
        '''The forward pass of the embedding head.'''
        x = self.pool(feat).flatten(1) #From [B, C, H, W] to [B, C].
        x = self.fc(x) #[B, out_dim]
        x = F.normalize(x, p=2, dim=1)
        return x
    
class FinalEncoder(nn.Module):
    '''Encoder + classifier head + embedding head.'''
    def __init__(self, num_classes=2, in_channels=64, hidden_dim=512, emb_dim=128):
        super().__init__()
        
        self.encoder = Encoder()
        self.classifier = ClassifierHead(in_channels=in_channels, hidden_dim=hidden_dim, num_classes=num_classes)
        self.embedder = EmbeddingHead(in_channels=in_channels, out_dim=emb_dim)

    def forward(self, x):
        '''The forward pass of the final model.'''
        features = self.encoder(x)
        logits = self.classifier(features)        
        emb = self.embedder(features)            
        return logits, emb

    def last_conv_layer(self):
        """Return the last Conv2d layer in the encoder for further Grad-CAM analysis."""
        last_conv = None

        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        
        return last_conv

class Autoencoder(nn.Module):
    '''A feed-forward neural network autoencoder with 4 convolutional layers in each part.'''
    def __init__(self):
        super(Autoencoder, self).__init__()

        #Encoder: compresses the input image to a lower-dimensional representation.
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1), #Dimensions: 512, 288, 8 (channels)
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), #Dimensions: 256, 144, 16
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), #Dimensions: 128, 72, 32
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        )
        
        #Decoder: reconstructs the image from the lower-dimensional latent representation.
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), #Input: 64x36, 64 channels
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  #Output: 1024x576, 3 channels
        nn.Sigmoid() #Output pixel values in range [0,1].
    )

    def forward(self,x):
        '''The forward pass of the autoencoder.'''
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded