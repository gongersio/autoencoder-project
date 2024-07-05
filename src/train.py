import torch.nn as nn
import torch.optim as optim
from model import FrameDataset, Autoencoder
from torch.utils.data import DataLoader

dataset = FrameDataset("./frameless") #CHANGE SOURCE DIRECTORY
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Autoencoder()
criterion = nn.MSELoss() #Loss Function (Mean Squared Error)
optimiser = optim.Adam(model.parameters(), lr=0.001)

#Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    for img in dataloader:
        #Get the reconstructed image by running the model on the input.
        output = model(img)
        # Compute the loss between the images
        loss = criterion(output, img)

        #Backward pass: compute gradient and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')