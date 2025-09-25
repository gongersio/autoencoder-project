import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from models import Encoder, FinalEncoder
from load_dataset import FrameDataset
from triplet_dataset import TripletDataset
from tqdm import tqdm

def load_dataset(file_path):
    '''Load a previously saved dataset from the specified file path.'''
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

def train_epoch(model, loader, optimizer, classification_criterion, triplet_criterion, device, triplet_weight=0.5):
    """Run a single training epoch using combined classification and triplet loss."""
    model.train()
    total_loss = 0.0

    for anchor_img, positive_img, negative_img, anchor_label in tqdm(loader, desc="Training"):
        #Move all data to the selected device.
        anchor_img = anchor_img.to(device)
        positive_img = positive_img.to(device)
        negative_img = negative_img.to(device)
        anchor_label = anchor_label.to(device)

        #Clear all the gradients.
        optimizer.zero_grad()

        #Forward pass for all three images.
        anchor_logits, anchor_emb = model(anchor_img)
        _, positive_emb = model(positive_img)
        _, negative_emb = model(negative_img)

        #Classification loss on the anchor image.
        class_loss = classification_criterion(anchor_logits, anchor_label)

        #Triplet loss on the embeddings.
        triplet_loss = triplet_criterion(anchor_emb, positive_emb, negative_emb)

        #Combine the losses with a weighting factor.
        loss = class_loss + (triplet_weight * triplet_loss)

        #Backward pass and update model weights.
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(loader)

def evaluate(model, loader, criterion, device):
    """Evaluates the model's classification performance on a validation dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            #Get the all the predicted class logits and count the number of correct predictions. 
            logits, _ = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss/len(loader)
    accuracy = 100 * correct/total
    return avg_loss, accuracy

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_PATH = "datasets/train_model.pkl"
    EVAL_DATASET_PATH = "datasets/eval_model.pkl"
    NUM_EPOCHS = 25
    LEARNING_RATE = 3e-4
    TRIPLET_WEIGHT = 0.3
    BATCH_SIZE = 32

    print(f"Using device: {DEVICE}")

    train_dataset = load_dataset(TRAIN_PATH)
    triplet_train_dataset = TripletDataset(train_dataset)
    train_loader = DataLoader(triplet_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    eval_dataset = load_dataset(EVAL_DATASET_PATH)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    #Initialize all the parts of the model.
    model = FinalEncoder(num_classes=2).to(DEVICE)

    classification_criterion = nn.CrossEntropyLoss()
    triplet_criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    #Main Training Loop
    best_eval_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, classification_criterion, triplet_criterion, DEVICE, TRIPLET_WEIGHT)
        eval_loss, eval_accuracy = evaluate(model, eval_loader, classification_criterion, DEVICE)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Evaluation Loss: {eval_loss:.4f}")
        print(f"Evaluation Accuracy: {eval_accuracy:.2f}%")

        scheduler.step(eval_loss)

        #Save the model if it has the best evaluation accuracy so far.
        if eval_accuracy > best_eval_accuracy:
            best_eval_accuracy = eval_accuracy
            torch.save(model.state_dict(), "models/model.pth")
            print("New best model saved.")

    print("\n--- Training Complete ---")
    print(f"Final best evaluation accuracy: {best_eval_accuracy:.2f}%")