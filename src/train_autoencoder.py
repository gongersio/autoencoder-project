import pickle
import os
from models import FinalEncoder, Autoencoder
from load_dataset import FrameDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

def load_dataset(file_path):
    '''Load a previously saved dataset from the specified file path.'''
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

def save_images(original, reconstructed, epoch, batch_idx, output_dir='comparison_grids'):
    '''Save a comparison of a selection (5 each) of both original and reconstructed frames.'''
    os.makedirs(output_dir, exist_ok=True)

    original_grid = torch.cat([original[i] for i in range(min(len(original), 5))], 2)
    reconstructed_grid = torch.cat([reconstructed[i] for i in range(min(len(reconstructed), 5))], 2)

    #Combine both the grids to get a side-by-side comparison.
    comparison_grid = torch.cat((original_grid, reconstructed_grid), 1)

    save_image(comparison_grid, f"{output_dir}/epoch_{epoch}_batch_{batch_idx}.png")

def train_autoencoder(model: Autoencoder, train_loader: DataLoader, eval_loader: DataLoader, criterion: nn.MSELoss, optimiser: optim.Adam, device, num_epochs=20):
    '''Train the autoencoder model for the specified number of epochs.'''
    for epoch in range(num_epochs):
        model.train() #Set the model to training mode.
        total_train_loss = 0.0

        for batch_idx, (batch_frames, _) in enumerate(train_loader):
            batch_frames = batch_frames.to(device)

            #Clear all the gradients.
            optimiser.zero_grad()

            #Perform a forward pass to get the reconstructed frames.
            _, reconstructed_frames = model(batch_frames)

            #Compute the loss between the original and reconstructed frames. 
            batch_loss: torch.Tensor = criterion(reconstructed_frames, batch_frames)

            #Perform a backward pass and update the model weights based on the loss.
            batch_loss.backward()
            optimiser.step()

            #Calculate the total epoch loss.
            total_train_loss += batch_loss.item() * batch_frames.size(0)

            #Save a comparison grid image for every second epoch.
            if epoch % 2 == 0 and batch_idx == 0:
                save_images(batch_frames.cpu(), reconstructed_frames.cpu(), epoch, batch_idx)

        #Calculate the average training loss per epoch.
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        model.eval() #Set the model to evaluation mode.
        total_eval_loss = 0.0

        with torch.no_grad():
            for batch_frames, _ in eval_loader:
                batch_frames = batch_frames.to(device)
                _, reconstructed_frames = model(batch_frames)
                batch_loss: torch.Tensor = criterion(batch_frames, reconstructed_frames)
                total_eval_loss += batch_loss.item() * batch_frames.size(0)

        #Calculate the average validation loss per epoch.
        avg_eval_loss = total_eval_loss / len(eval_loader.dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_eval_loss:.4f}')


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    #Load the training and evaluation datasets.
    train_dataset = load_dataset("datasets/train_model.pkl")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    eval_dataset = load_dataset("datasets/eval_model.pkl")
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    #Initialise the Autoencoder using the pretrained encoder weights from the FinalEncoder.
    model = Autoencoder().to(DEVICE)
    trained_encoder_model = FinalEncoder(num_classes=2)
    trained_encoder_model.load_state_dict(torch.load("models/model_85_NEW.pth"))
    model.encoder.load_state_dict(trained_encoder_model.encoder.encoder.state_dict())

    #Freeze the encoder parameters to prevent them from being updated during training.
    for param in model.encoder.parameters():
        param.requires_grad = False

    #Define the loss function and optimiser that only trains the decoder.
    criterion = nn.MSELoss() #Loss Function (Mean Squared Error)
    optimiser = optim.Adam(model.decoder.parameters(), lr=0.001)

    train_autoencoder(model, train_loader, eval_loader, criterion, optimiser, DEVICE)

    #Save the autoencoder model.
    torch.save(model.state_dict(), "models/autoencoder_NEW.pth")
    print("Model saved.")