import torch
import os
import pickle

from collections import Counter, defaultdict
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from models import FinalEncoder
from load_dataset import FrameDataset
from numpy import mean

def load_model(model_path, device):
    """Loads the trained model for video predictions."""
    model = FinalEncoder(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def predict(model, pkl_path, device, class_names):
    """Loads a pre-processed dataset and performs majority vote classification from the video frames."""

    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    #Gather all frame level predictions.
    video_predictions = defaultdict(list)
    video_true_labels = {}

    total_frames = 0
    correct_frames = 0

    with torch.no_grad():
        #The dataset returns (frame, label, video_id).
        for frames, labels, video_ids in tqdm(loader, desc="Predicting"):
            frames = frames.to(device)
            
            logits, _ = model(frames)

            #Convert logits to probabilities to get a confidence score.
            probabilities = torch.softmax(logits, dim=1)
            scores, preds = torch.max(probabilities, 1)

            for i in range(len(video_ids)):
                vid_id = video_ids[i].item()
                true_label = labels[i].item()
                prediction = preds[i].item()
                score = scores[i].item()

                video_predictions[vid_id].append((prediction, score))

                if vid_id not in video_true_labels:
                    video_true_labels[vid_id] = true_label

                #Calculate frame-level accuracy.
                total_frames += 1
                if prediction == true_label:
                    correct_frames += 1

    #Perform video-level classification based on average frame logits and calculate overall accuracy.
    correct_videos = 0
    total_videos = len(video_true_labels)

    for vid_id, frame_preds in sorted(video_predictions.items()):
        true_label = video_true_labels[vid_id]
        predictions = [p for p, s in frame_preds]

        #Majority voting for final video prediction.
        vote_counts = Counter(predictions)
        final_prediction, _ = vote_counts.most_common(1)[0]

        #For each frame, get the probability assigned to the winning class.
        confidence_scores = []
        for pred_class, score in frame_preds:
            if pred_class == final_prediction: confidence_scores.append(score)

            else: confidence_scores.append(1 - score) #If the frame predicted a different class, use 1 - score.

        video_confidence = mean(confidence_scores)

        is_correct = "CORRECT" if final_prediction == true_label else "INCORRECT"
        frame_preds_str = [f"'{class_names[p]}' ({s:.2f})" for p, s in frame_preds]

        print(
            f"Video ID {vid_id}: Frame Preds=[{', '.join(frame_preds_str)}], "
            f"Final Pred='{class_names[final_prediction]}' (Conf: {video_confidence:.2f}), "
            f"True Label='{class_names[true_label]}' --> {is_correct}"
        )

        if final_prediction == true_label:
            correct_videos += 1
            
    video_accuracy = 100 * (correct_videos/total_videos)
    frame_accuracy = 100 * (correct_frames / total_frames)

    return video_accuracy, frame_accuracy, correct_videos, total_videos, correct_frames, total_frames

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "models/model_85_NEW.pth"
    DATASET_PKL = "datasets/test_model.pkl" 
    
    class_names = {0: "Traditional", 1: "Immersive"}

    #Load the trained model.
    model = load_model(MODEL_PATH, DEVICE)
    print(f"Model loaded from {MODEL_PATH}. Using device: {DEVICE}.")

    #Get the final video-level accuracy.
    video_accuracy, frame_accuracy, correct_videos, total_videos, correct_frames, total_frames = predict(model, DATASET_PKL, DEVICE, class_names)

    print("\n--- Final Test Results ---")
    print(f"Frame-Level Accuracy: {frame_accuracy:.2f}% ({correct_frames}/{total_frames})")
    print(f"Video-Level Accuracy: {video_accuracy:.2f}% ({correct_videos}/{total_videos})")
