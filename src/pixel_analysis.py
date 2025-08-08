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
from sklearn.cluster import KMeans

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
    '''Normalize the pixel values in the latent map for better visualisation.'''
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def generate_image_heatmaps(loader: DataLoader, encoder: Autoencoder, output_dir='heatmaps', topk=4):
    '''Generate heatmaps for each image per class by calculating average activation across all channels. Also find the top-k most different channels between the two classes.'''
    to_img = transforms.ToPILImage()
    global_id = 0

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

                #Average activation for each pixel in the image across all 128 channels.
                avg_map = features.mean(dim=0)
                avg_np = (avg_map.numpy()*255).round().astype(np.uint8) #Scale 1-255, as rounded integers.   

                #Calculate the standard deviation of the top and bottom 8 rows combined.
                rows = np.concatenate([avg_np[:8], avg_np[-8:]], axis=0)

                #Calculate the standard deviation of the left and right 3 columns combined.
                cols = np.concatenate([avg_np[:, :3], avg_np[:, -3:]], axis=1)

                #Remove images with bars on the sides from resizing them.
                if np.std(rows) < 2 or np.std(cols) < 4:
                    print(f"Skipping image {global_id} (label {label})")
                    global_id += 1
                    continue

                #Convert the heatmaps to images
                fname = f'img_{global_id}.png'

                avg_img = to_img(normalize(avg_map))
                avg_img.save(os.path.join(output_dir, f'class_{label}', 'avg', fname))

                #Accumulate per-class channel sums.
                if label not in channel_sums:
                    channel_sums[label] = torch.zeros_like(features)
                    channel_counts[label] = 0

                channel_sums[label] += features
                channel_counts[label] += 1

                global_id += 1

    labels = sorted(channel_sums.keys())

    #Calculate the average activation map for each channel per class.
    avg_channels_0 = channel_sums[labels[0]] / channel_counts[labels[0]]
    avg_channels_1 = channel_sums[labels[1]] / channel_counts[labels[1]]

    #Calculate the difference between each channel in both classes.
    difference_map = torch.abs(avg_channels_0 - avg_channels_1)

    #Score each channel by the total difference between the two classes.
    channel_scores = torch.sum(difference_map, dim=(1, 2))

    #Get the indices of the top-k most different channels
    topk_indices = torch.topk(channel_scores, topk).indices

    for rank, ch_idx in enumerate(topk_indices):
        ch_map_0 = avg_channels_0[ch_idx]
        ch_map_1 = avg_channels_1[ch_idx]

        ch_img_0 = to_img(normalize(ch_map_0))
        ch_img_1 = to_img(normalize(ch_map_1))

        path_0 = os.path.join(output_dir, f'class_{labels[0]}', 'topk')
        path_1 = os.path.join(output_dir, f'class_{labels[1]}', 'topk')

        ch_img_0.save(os.path.join(path_0, f"top{rank+1}_ch{ch_idx.item()}.png"))
        ch_img_1.save(os.path.join(path_1, f"top{rank+1}_ch{ch_idx.item()}.png"))

def generate_class_heatmaps(dir, avg_output, cluster_output, pattern="*.png", clusters=3):
    '''Generate an average heatmap across all images in a certain class. Also create k clusters of similar heatmaps for each class.'''
    files = list(dir.glob(pattern))
    accumulator = None

    img_maps = []
    flat_maps = []

    for f in files:
        img = Image.open(f)
        arr = np.asarray(img, dtype=np.float32)

        img_maps.append(arr)
        flat_maps.append(arr.flatten())

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

    Image.fromarray(stretch).save(avg_output)

    X = np.stack(flat_maps) #(N, height x width)

    #Run K-Means clustering to group similar maps.
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    labels = kmeans.fit_predict(X)

    #Average all maps per cluster.
    for cluster_id in range(clusters):
        cluster_maps = [img_maps[i] for i in range(len(labels)) if labels[i] == cluster_id]

        cluster_avg = np.mean(cluster_maps, axis=0)
        mn, mx = cluster_avg.min(), cluster_avg.max()

        #Normalise the array for better visualisation.
        if mx > mn:
            stretched = ((cluster_avg - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            stretched = cluster_avg.clip(0, 255).astype(np.uint8)

        out_path = os.path.join(cluster_output, f"cluster_{cluster_id}.png")
        Image.fromarray(stretched).save(out_path)

def generate_difference_maps(file1, file2):
    '''Add docstring.'''
    arr1 = np.array(Image.open(file1), dtype=np.float32)
    arr2 = np.array(Image.open(file2), dtype=np.float32)

    diff = arr2 - arr1
    min_diff, max_diff = diff.min(), diff.max()

    #Normalise the array for better visualisation. Whiter pixels indicate larger differences, where Class 1 has higher activations.
    if max_diff > min_diff:
        diff_normalized = ((diff - min_diff) / (max_diff - min_diff) * 255).astype(np.uint8)

    else:
        diff_normalized = diff.clip(0, 255).astype(np.uint8)
    
    Image.fromarray(diff_normalized).save("diff.png")

if __name__ == "__main__":
    encoder = load_encoder("models/autoencoder.pth")

    dataset = load_dataset("datasets/train_classifier.pkl")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    generate_image_heatmaps(loader, encoder)

    generate_class_heatmaps(Path("heatmaps/class_0/avg"), "average0.png", Path("heatmaps/class_0/clusters"))
    generate_class_heatmaps(Path("heatmaps/class_1/avg"), "average1.png", Path("heatmaps/class_1/clusters"))

    generate_difference_maps("average0.png", "average1.png")