import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from models import Autoencoder
from PIL import Image
from torchvision import transforms

def load_encoder(file_path):
    '''Load a pre-trained autoencoder from the specified file path.'''
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(file_path))
    autoencoder.eval()
    
    #Extract and return only the encoder part of the model.
    return autoencoder.encoder

def load_decoder(file_path):
    '''Load a pre-trained autoencoder from the specified file path.'''
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(file_path))
    autoencoder.eval()
    
    #Extract and return only the encoder part of the model.
    return autoencoder.decoder

def load_image(path, output_path):
    image = Image.open(path)
    image = image.resize((288,512))

    convert_tensor = transforms.ToTensor()
    image = convert_tensor(image)
    image = image.unsqueeze(0)

    features: torch.Tensor = encoder(image)

    rec = decoder(features)
    rec = rec.squeeze(0)
    to_image = transforms.ToPILImage()
    reconstructed_image = to_image(rec)

    reconstructed_image.save("reconstructed_img_1187.jpg")

    features = features.squeeze(0)

    #Visualise all channels.
    os.makedirs("outputs", exist_ok=True)  # Create output directory if it doesn't exist
    for i in range(features.size(0)):
        channel = features[i]  # Get the i-th channel
        normalized_channel = (channel - channel.min()) / (channel.max() - channel.min())  # Normalize the channel
        
        channel_image = to_image(normalized_channel)
        channel_image.save(os.path.join("outputs", f'channel_{i+1}.jpg'))  # Save each channel as an image

    #Average pixel activation across all channels.
    average_activation = features.mean(dim=0) 
    normalized_activation = (average_activation - average_activation.min()) / (average_activation.max() - average_activation.min())
    
    global sum_of_squares
    global accumulated_activations

    if sum_of_squares is None:
        sum_of_squares = torch.zeros_like(normalized_activation)
    
    if accumulated_activations is None:
        accumulated_activations = torch.zeros_like(normalized_activation)

    sum_of_squares += normalized_activation ** 2
    accumulated_activations += normalized_activation

    to_image = transforms.ToPILImage()
    activation_image = to_image(normalized_activation)

    activation_image.save(output_path)

    overall_activation = average_activation.mean().item()

    return overall_activation #overall average across image - shows feature density of image

encoder = load_encoder("autoencoder.pth")
decoder = load_decoder("autoencoder.pth")

class_activations = []
accumulated_activations = None
sum_activations = None
sum_of_squares = None

img_1 = load_image("img_0762.jpg", "latent_img_0762.jpg")
class_activations.append(img_1)
img_2 = load_image("img_1187.jpg", "latent_img_1187.jpg")
class_activations.append(img_2)
img_3 = load_image("img_0000.jpg", "latent_img_0000.jpg")
class_activations.append(img_3)

class_activations = np.array(class_activations)
mean = np.mean(class_activations) #shows feature density of class - maybe frameless has more features?
var = np.var(class_activations) #lower value means class is homogenous - similar features.
print(mean, var)

avg_pixel_activation = accumulated_activations/3
to_image = transforms.ToPILImage()
avg_img = to_image(avg_pixel_activation)
avg_img.save("average_pixels.jpg") #average activation of each pixel in the class - show where features occur.

variance_activation = (sum_of_squares / 3) - (avg_pixel_activation ** 2)
to_pil = transforms.ToPILImage()
range_activation_image = to_pil(variance_activation)
range_activation_image.save("pixel_var_class.jpg") #show where 'unique' features occur - clustering and manual analysis to identify.
#create difference map to show which pixels are activated more per class.





