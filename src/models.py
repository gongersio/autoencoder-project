import torch.nn as nn

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
        )
        
        #Decoder: reconstructs the image from the lower-dimensional latent representation.
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), #Dimensions: 256, 144, 16 
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1), #Dimensions: 512, 288, 8
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1), #Dimensions: 1024, 576, 3
        nn.Sigmoid() #Output pixel values in range [0,1].
        )

    def forward(self,x):
        '''The forward pass of the autoencoder.'''
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded