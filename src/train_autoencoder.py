import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from models import Autoencoder
from load_dataset import FrameDataset
from torch.utils.data import DataLoader

def load_dataset(file_path):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

def save_images(original, reconstructed, epoch, batch_idx, output_dir='output_images'):
    os.makedirs(output_dir, exist_ok=True)

    original_grid = torch.cat([original[i] for i in range(min(len(original), 8))], 2)
    reconstructed_grid = torch.cat([reconstructed[i] for i in range(min(len(reconstructed), 8))], 2)

    comparison = torch.cat((original_grid, reconstructed_grid), 1)

    save_image(comparison, f"{output_dir}/epoch_{epoch}_batch_{batch_idx}.png")

def train_autoencoder(model: Autoencoder, train_loader: DataLoader, eval_loader: DataLoader, criterion: nn.MSELoss, optimiser: optim.Adam, num_epochs=20):
    for epoch in range(num_epochs):
        model.train() #Set the model to training mode.
        total_epoch_loss = 0.0

        for batch_idx, (frames, _) in enumerate(train_loader):
            #Clear all the gradients.
            optimiser.zero_grad()

            #Perform a forward pass to get the reconstructed image.
            _, outputs = model(frames)

            #Compute the loss between the original and reconstructed images. 
            batch_loss: torch.Tensor = criterion(outputs, frames)

            #Perform the backward pass and update the model weights.
            batch_loss.backward()
            optimiser.step()

            #Calculate the total epoch loss.
            total_epoch_loss += batch_loss.item() * frames.size(0)

            if epoch % 4 == 0 and batch_idx == 0:
                save_images(frames, outputs, epoch, batch_idx)

        avg_train_loss = total_epoch_loss / len(train_loader.dataset)

        model.eval() #Set the model to evaluation mode.
        total_eval_loss = 0.0

        with torch.no_grad():
            for frames, _ in eval_loader:
                _, outputs = model(frames)
                loss: torch.Tensor = criterion(outputs, frames)
                total_eval_loss += loss.item() * frames.size(0)

        avg_eval_loss = total_eval_loss / len(eval_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_eval_loss:.4f}')

train_dataset = load_dataset("combined_train_dataset.pkl")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

eval_dataset = load_dataset("combined_eval_dataset.pkl")
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

#Initialise the criterion, model, and optimiser.
model = Autoencoder()
criterion = nn.MSELoss() #Loss Function (Mean Squared Error)
optimiser = optim.Adam(model.parameters(), lr=0.001)

#Train the autoencoder.
train_autoencoder(model, train_loader, eval_loader, criterion, optimiser)
torch.save(model.state_dict(), "autoencoder.pth")
print("Model saved.")