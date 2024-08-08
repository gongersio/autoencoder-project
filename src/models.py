import torch.nn as nn

class Autoencoder(nn.Module):
    '''A feed-forward neural network autoencoder with 4 convolutional layers in each part.'''
    def __init__(self):
        super(Autoencoder, self).__init__()

        #Encoder: compresses the input image to a lower-dimensional representation.
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), #Dimensions: 128, 72, 16 (RGB channels)
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), #Dimensions: 64, 36, 32
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #Dimensions: 32, 18, 64
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), #Dimensions: 16, 9, 128
        nn.BatchNorm2d(128),
        nn.ReLU()
        )
        
        #Decoder: reconstructs the image from the lower-dimensional latent representation
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), #Dimensions: 32, 18, 64 
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), #Dimensions: 64, 36, 32
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), #Dimensions: 128, 72, 16
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # Dimension: 256, 144, 3
        nn.Sigmoid() #Output pixel values in range [0,1].
        )

    def forward(self,x):
        '''The forward pass of the autoencoder.'''
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
class Classifier(nn.Module):
    '''A feed-forward neural network classifier with one hidden layer.

        Args:
        input_dim: Dimensionality of input features (i.e. size of the extracted feature vector).
        hidden_dim (int): Dimensionality (number of neurons) in the hidden layer.
        output_dim (int): Number of distinct output classes (e.g. non-immersive vs immersive).
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()

        #Linear transformation from the input features to the hidden layer.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        #Linear transformation from the hidden layer to output class probabilities.
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''The forward pass of the classifier.'''
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.softmax(output)
        return output