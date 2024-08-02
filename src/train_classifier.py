import torch
import torch.nn as nn
import torch.optim as optim
from models import Autoencoder, Classifier
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_encoder(file_path):
    '''Load a pre-trained autoencoder from the specified file path.'''
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(file_path))
    autoencoder.eval()
    
    #Extract the encoder part of the model.
    encoder = autoencoder.encoder
    return encoder

def extract_features(encoder, dataloader):
    '''Extract features from the compressed representation outputted by the encoder.'''
    features = []
    labels = []

    with torch.no_grad():
        for batch_frames, batch_labels in dataloader:
            batch_features = encoder(batch_frames)
            features.append(batch_features.view(batch_features.size(0), -1))  #Flatten the features.
            labels.append(batch_labels)
    
    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels

def train_classifier(features, labels, input_dim, hidden_dim, output_dim, num_epochs=20):
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    model = Classifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0.0
        
        for batch_features, batch_labels in dataloader:
            optimiser.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimiser.step()
            total_epoch_loss += loss.item() * batch_features.size(0)
        
        avg_epoch_loss = total_epoch_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

    return model

def evaluate_classifier(model, test_features, test_labels):
    model.eval()

    with torch.no_grad():
        outputs = model(test_features)
        _, predicted = torch.max(outputs, 1)

        accuracy = accuracy_score(test_labels, predicted)
        precision = precision_score(test_labels, predicted, average='binary')
        recall = recall_score(test_labels, predicted, average='binary')
        f1 = f1_score(test_labels, predicted, average='binary')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

encoder = load_encoder("autoencoder.pth")

