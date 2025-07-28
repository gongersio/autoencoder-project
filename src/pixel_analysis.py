import pickle
import os
import torch
import numpy as np

from load_dataset import FrameDataset
from models import Autoencoder
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

def load_dataset(file_path):
    '''Load a previously saved dataset from the specified file path.'''
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

def load_encoder(file_path):
    '''Load a pre-trained autoencoder from the specified file path.'''
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(file_path))
    autoencoder.eval()
    
    #Extract and return only the encoder part of the model.
    return autoencoder.encoder

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def generate_image_heatmaps(loader: DataLoader, encoder: Autoencoder, output_dir='heatmaps', topk=7):
    to_img = transforms.ToPILImage()
    global_id = 0

    num_channels = 128
    channel_sums = {} #Label -> (channels, height, width)
    channel_counts = {}

    with torch.no_grad():
        for batch_imgs, batch_labels in loader:
            #Get the latent representations of all the images within the batch.
            latent = encoder(batch_imgs) #(batch, channels, height, width)

            #Iterate through each image in the batch.
            for i in range(latent.size(0)):
                features = latent[i] #(channels, height, width)
                label = batch_labels[i].item()

                #Average activation for each pixel across all 128 channels.
                avg_map = features.mean(dim=0)
                avg_np = (avg_map.numpy()*255).round().astype(np.uint8) #Scale 1-255, as rounded integers.    

                #Maximum activation for each pixel across all 128 channels.
                max_map, _ = features.max(dim=0)

                #Images with too few unique values are likely to be anomalies.
                if len(np.unique(avg_np)) < 20:
                    print(f"Skipping image {global_id} (label {label}): too few unique values")
                    continue

                #Convert the heatmaps to images
                fname = f'img_{global_id}.png'

                avg_img = to_img(normalize(avg_map))
                avg_img.save(os.path.join(output_dir, f'class_{label}', 'avg', fname))

                max_img = to_img(normalize(max_map))
                max_img.save(os.path.join(output_dir, f'class_{label}', 'max', fname))

                #Accumulate per-class channel sums.
                if label not in channel_sums:
                    channel_sums[label] = torch.zeros_like(features)
                    channel_counts[label] = 0

                channel_sums[label] += features
                channel_counts[label] += 1

                global_id += 1
    
    #Calculate the average activation in each individual channel for each class.
    for label, summed in channel_sums.items():
        count = channel_counts[label]
        avg_channels = summed / count

        #Score each channel by standard deviation to find spatially diverse channels.
        channel_scores = avg_channels.std(dim=(1, 2))
        topk_indices = torch.topk(channel_scores, topk).indices

        path = os.path.join(output_dir, f'class_{label}', 'topk')

        for rank, ch_idx in enumerate(topk_indices):
            ch_map = avg_channels[ch_idx]
            ch_img = to_img(normalize(ch_map))
            ch_img.save(os.path.join(path, f"top{rank+1}_ch{ch_idx.item()}.png"))

def generate_class_heatmaps(dir, output, pattern="*.png"):
    files = list(dir.glob(pattern))
    accumulator = None

    for f in files:
        img = Image.open(f)
        arr = np.asarray(img, dtype=np.float32)

        if accumulator is None:
            accumulator = np.zeros_like(arr, dtype=np.float32)

        accumulator += arr #Update the running sum.
    
    mean_arr = accumulator / len(files)
    mn, mx = mean_arr.min(), mean_arr.max()

    #Normalise the array for better visualisation.
    if mx > mn:
        stretch = ((mean_arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
    else:
        stretch = mean_arr.clip(0, 255).astype(np.uint8)

    Image.fromarray(stretch).save(output) 

def generate_difference_maps(file1, file2):
    arr1 = np.array(Image.open(file1), dtype=np.float32)
    arr2 = np.array(Image.open(file2), dtype=np.float32)

    diff = arr2 - arr1
    min_diff, max_diff = diff.min(), diff.max()

    #Normalise the array for better visualisation, centering the values at 128 to show contrast.
    if max_diff > min_diff:
        diff_normalized = ((diff - min_diff) / (max_diff - min_diff) * 255).astype(np.uint8)

    #Whiter pixels indicate larger differences.
    else:
        diff_normalized = diff.clip(0, 255).astype(np.uint8)
    Image.fromarray(diff_normalized).save("diff.png")

if __name__ == "__main__":
    encoder = load_encoder("models/autoencoder.pth")

    dataset = load_dataset("datasets/train_classifier.pkl")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    generate_image_heatmaps(loader, encoder)

    #generate_class_heatmaps(Path("heatmaps/class_0/avg"), "average0.png")
    #generate_class_heatmaps(Path("heatmaps/class_1/avg"), "average1.png")

    #generate_difference_maps("average0.png", "average1.png")