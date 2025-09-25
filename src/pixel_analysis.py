import pickle
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import shutil

from tqdm import tqdm
from PIL import Image
from load_dataset import FrameDataset
from models import FinalEncoder, Autoencoder
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.utils import save_image
from scipy.spatial.distance import cdist

def load_dataset(file_path):
    '''Load a previously saved dataset from the specified file path.'''
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

def load_model(file_path, device):
    '''Load a pre-trained model from the specified file path.'''
    model = FinalEncoder().to(device)
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.eval()
    
    #Extract and return the model.
    return model

def normalize(x, min_val, max_val):
    '''Normalizes the pixel values in the latent map for better visualisation.'''
    return (x - min_val) / (max_val - min_val + 1e-8)

def create_colour_coded_difference(diff_path: str, inverse_diff_path: str, output_path: str):
    """Combines the difference and inverse difference images into a single red/blue visualisation. Red = Class 1 features, Blue = Class 0 features."""
    img1 = Image.open(diff_path).convert("RGB")
    img0 = Image.open(inverse_diff_path).convert("RGB")

    #Extract the red channel from the Class 1 difference image (map1 - map0).
    #This channel represents features unique to Class 1.
    r, _, _ = img1.split()

    #Extract the blue channel from the Class 0 difference image (map0 - map1).
    #This channel represents features unique to Class 0.
    _, _, b = img0.split()
    
    #Create a blank green channel.
    g = Image.new("L", img1.size, 0)

    #Merge the channels into a new color image.
    #Red = Class 1 features, Blue = Class 0 features.
    composite_image = Image.merge("RGB", (r, g, b))
    
    composite_image.save(output_path)

def find_medoids(data_points, output_dir:str, num_medoids=5):
    """Finds and saves the medoid (most central point) of a given class."""
    os.makedirs(output_dir, exist_ok=True)

    #Find the n data points (medoids) closest to the centroid of the embeddings.
    embeddings = np.array([d['embedding'] for d in data_points])
    centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    distances = cdist(embeddings, centroid)
    top_n_indices = np.argsort(distances, axis=0)[:num_medoids]

    medoid_image_paths = []
    for rank, idx in enumerate(top_n_indices.flatten()):
        medoid_data = data_points[idx]
        
        source_path = medoid_data['image_path']
        dest_filename = f"medoid_rank_{rank+1}.png"
        dest_path = os.path.join(output_dir, dest_filename)

        shutil.copy(source_path, dest_path)
        medoid_image_paths.append(dest_path)
    
    #Create a composite images of all the medoids side by side.
    medoid_images = [Image.open(p) for p in medoid_image_paths]
    width, height = medoid_images[0].size

    #Create a new blank image for a horizontal grid.
    grid_image = Image.new('RGB', (width * len(medoid_images), height))
    
    #Paste each image into the grid
    for i, img in enumerate(medoid_images):
        grid_image.paste(img, (i * width, 0))
        
    grid_path = os.path.join(output_dir, f"top_{num_medoids}_grid.png")
    grid_image.save(grid_path)

def process_dataset(loader: DataLoader, model: FinalEncoder, device, output_path='datasets/processed_data.pkl', image_dir='images'):
    '''Processes the entire dataset through the model and save the outputs.'''
    os.makedirs(image_dir, exist_ok=True)
    processed_data = []
    img_counter = 0

    with torch.no_grad():
        for data_batch in tqdm(loader, desc="Processing Batches"):
            images, labels = data_batch[0].to(device), data_batch[1]
            labels_np = labels.numpy()

            #Get the latent representations of all the images within the batch.
            latent = model.encoder(images) #(batch, channels, height, width)
            latent_np = latent.cpu().numpy()

            _, embeddings = model(images)
            embeddings_np = embeddings.cpu().numpy()

            #Iterate through each image in the batch.
            for i in range(len(images)):
                image_filename = f"image_{img_counter}.png"
                image_path = os.path.join(image_dir, image_filename)
                save_image(images[i], image_path)

                map = latent_np[i] #(channels, height, width)

                data_point = {
                    'image_path': image_path,
                    'latent_map': map,
                    'embedding': embeddings_np[i],
                    'label': labels_np[i],
                }

                processed_data.append(data_point)
                img_counter += 1
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)

def generate_class_plots(data_path: str, output_dir: str):
    """Generates class separation plots using PCA, t-SNE, and UMAP."""
    os.makedirs(output_dir, exist_ok=True)

    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)
    
    embeddings = np.array([d['embedding'] for d in processed_data])
    labels = np.array([d['label'] for d in processed_data])

    #PCA Plot
    print("Generating PCA plot.")
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], hue=labels, palette="viridis", legend="full", alpha=0.7)

    plt.title('PCA Visualization of Embedding Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    pca_path = os.path.join(output_dir, 'pca_separation.png')
    plt.savefig(pca_path, dpi=300)
    plt.close()

    #t-SNE Plot
    print("Generating t-SNE plot.")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette="viridis", legend="full", alpha=0.7)

    plt.title('t-SNE Visualization of Embedding Space')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    tsne_path = os.path.join(output_dir, 'tsne_separation.png')
    plt.savefig(tsne_path, dpi=300)
    plt.close()

    #UMAP Plot
    print("Generating UMAP plot.")
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_results = reducer.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=umap_results[:, 0], y=umap_results[:, 1], hue=labels, palette="viridis", legend="full", alpha=0.7)
    plt.title('UMAP Visualization of Embedding Space')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    umap_path = os.path.join(output_dir, 'umap_separation.png')
    plt.savefig(umap_path, dpi=300)
    plt.close()

def generate_class_averages(data_path: str, autoencoder_path: str, output_dir: str):
    """Generates an average image 'prototype' for each class, and a difference heatmap between the two classes. Also finds and saves the medoids (most representative images) for each class."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(autoencoder_path))
    model.eval()

    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)
    
    #Separate data points by class.
    data_class_0 = [d for d in processed_data if d['label'] == 0]
    data_class_1 = [d for d in processed_data if d['label'] == 1]

    #Separate the latent maps by class.
    maps_class_0 = [torch.from_numpy(d['latent_map']) for d in processed_data if d['label'] == 0]
    maps_class_1 = [torch.from_numpy(d['latent_map']) for d in processed_data if d['label'] == 1]

    #Calculate the average "prototype" map for each class.
    if maps_class_0:
        prototype_map_0 = torch.stack(maps_class_0).mean(dim=0)

        with torch.no_grad():
            #Add a batch dimension and move to device for the decoder.
            reconstructed_prototype_0 = model.decoder(prototype_map_0.unsqueeze(0).to(device))
            save_image(reconstructed_prototype_0, os.path.join(output_dir, 'avg_prototype_0.png'))

    if maps_class_1:
        prototype_map_1 = torch.stack(maps_class_1).mean(dim=0)

        with torch.no_grad():
            #Add a batch dimension and move to device for the decoder.
            reconstructed_prototype_1 = model.decoder(prototype_map_1.unsqueeze(0).to(device))
            save_image(reconstructed_prototype_1, os.path.join(output_dir, 'avg_prototype_1.png'))

    #Calculate the difference and inverse difference maps between the two class prototypes.
    with torch.no_grad():
        diff_map = prototype_map_1 - prototype_map_0
        reconstructed_diff = model.decoder(diff_map.unsqueeze(0).to(device))
        save_image(reconstructed_diff, os.path.join(output_dir, 'difference_map.png'))

        inverse_diff_map = prototype_map_0 - prototype_map_1
        reconstructed_inverse_diff = model.decoder(inverse_diff_map.unsqueeze(0).to(device))
        save_image(reconstructed_inverse_diff, os.path.join(output_dir, 'inverse_difference_map.png'))

    create_colour_coded_difference(
        diff_path=os.path.join(output_dir, 'difference_map.png'),
        inverse_diff_path=os.path.join(output_dir, 'inverse_difference_map.png'),
        output_path=os.path.join(output_dir, 'colour_coded_difference.png')
    )

    os.remove(os.path.join(output_dir, 'difference_map.png'))
    os.remove(os.path.join(output_dir, 'inverse_difference_map.png'))

    find_medoids(data_class_0, os.path.join(output_dir, 'class_0'), num_medoids=7)
    find_medoids(data_class_1, os.path.join(output_dir, 'class_1'), num_medoids=7)

def generate_topk_channels(data_path: str, autoencoder_path: str, output_dir: str, topk=7):
    '''Finds the top-k most discriminative channels between two classes and visualises the difference between them.'''
    for subdir in ['class0', 'class1', 'diff']:
        path = os.path.join(output_dir, subdir)
        os.makedirs(path, exist_ok=True)

        #Clear existing files in the directory.
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(autoencoder_path))
    model.eval()

    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)

    all_maps = [torch.from_numpy(d['latent_map']) for d in processed_data]
    baseline_map = torch.stack(all_maps).mean(dim=0).to(device)

    #Separate the latent maps by class.
    latent_maps_0 = np.array([d['latent_map'] for d in processed_data if d['label'] == 0])
    latent_maps_1 = np.array([d['latent_map'] for d in processed_data if d['label'] == 1])

    #Calculate the mean latent map for each class.
    avg_latent_map_0 = np.mean(latent_maps_0, axis=0)
    avg_latent_map_1 = np.mean(latent_maps_1, axis=0)

    #Calculate the difference between each channel in both classes.
    difference_map = np.abs(avg_latent_map_1 - avg_latent_map_0)

    #Score each channel by the total difference between the two classes.
    channel_scores = np.sum(difference_map, axis=(1, 2))

    #Get the indices of the top k most discriminative channels.
    topk_indices = np.argsort(channel_scores)[::-1][:topk]

    with torch.no_grad():
        #Start by creating composite maps that contain the average activations for all channels.
        composite_map_0 = baseline_map.clone()
        composite_map_1 = baseline_map.clone()

        #Replace the top-k channels in the composite maps with the average activations from their respective classes.
        for idx in topk_indices:
            composite_map_0[idx] = torch.from_numpy(avg_latent_map_0[idx]).to(device)
            composite_map_1[idx] = torch.from_numpy(avg_latent_map_1[idx]).to(device)

        decoded_composite_0 = model.decoder(composite_map_0.unsqueeze(0))
        save_image(decoded_composite_0, os.path.join(output_dir, "class0/composite_0.png"))

        decoded_composite_1 = model.decoder(composite_map_1.unsqueeze(0))
        save_image(decoded_composite_1, os.path.join(output_dir, "topk/class1/composite_1.png"))

        diff_map = composite_map_1 - composite_map_0
        decoded_diff = model.decoder(diff_map.unsqueeze(0))
        diff_output_path = os.path.join(output_dir, "diff/composite_diff.png")
        save_image(decoded_diff, diff_output_path)

        inv_diff_map = composite_map_0 - composite_map_1
        decoded_inv_diff = model.decoder(inv_diff_map.unsqueeze(0))
        inv_diff_output_path = os.path.join(output_dir, "diff/composite_inv_diff.png")
        save_image(decoded_inv_diff, inv_diff_output_path)

        create_colour_coded_difference(
            diff_path=diff_output_path,
            inverse_diff_path=inv_diff_output_path,
            output_path=os.path.join(output_dir, "diff/composite_colour_diff.png")
        )

        os.remove(os.path.join(output_dir, f"diff/composite_diff.png"))
        os.remove(os.path.join(output_dir, f"diff/composite_inv_diff.png"))

def generate_clusters(data_path: str, autoencoder_path: str, output_dir: str, clusters=6):
    """Performs K-Means clustering on image embeddings from each class and visualises the medoids of each cluster."""
    os.makedirs(output_dir, exist_ok=True)

    #Clear all existing subdirectories and files in the output directory.
    for subdir in os.listdir(output_dir):
        full_path = os.path.join(output_dir, subdir)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load(autoencoder_path))
    autoencoder.eval()

    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)

    #Separate data points by class.
    data_class_0 = [d for d in processed_data if d['label'] == 0]
    data_class_1 = [d for d in processed_data if d['label'] == 1]

    for class_label, class_data in enumerate([data_class_0, data_class_1]):
        class_embeddings = np.array([d['embedding'] for d in class_data])

        #Perform k-means clustering on the embeddings for this class.
        kmeans = KMeans(n_clusters=clusters, random_state=42)
        kmeans.fit(class_embeddings)
        cluster_labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        cluster_counts = np.bincount(cluster_labels)
        print(f"Class {class_label} - cluster distribution:")

        for i, count in enumerate(cluster_counts):
            print(f"  - Cluster {i}: {count} members")

        #Find and visualise the medoids (most central points) of each cluster.
        for i in range(clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_data = [class_data[idx] for idx in cluster_indices]
            find_medoids(cluster_data, os.path.join(output_dir, f'class_{class_label}_cluster_{i}'), num_medoids=7)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("models/model_83_NEW.pth", device)

    dataset = load_dataset("datasets/analyse_model.pkl")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    #process_dataset(loader, model, device)
    #generate_class_plots("datasets/processed_data.pkl", "heatmaps/plots")
    #generate_class_averages("datasets/processed_data.pkl", "models/autoencoder_NEW.pth", "heatmaps/average")
    #generate_topk_channels("datasets/processed_data.pkl", "models/autoencoder_NEW.pth", "heatmaps/topk")
    generate_clusters("datasets/processed_data.pkl", "models/autoencoder_NEW.pth", "heatmaps/clusters")
