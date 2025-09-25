import os
import cv2
import re
import torch
import torchvision
from collections import defaultdict

def create_video_frame_grids(input_folder, output_folder, margin=2):
    """Scans the input folder for images and creates a grid for each video's frames."""

    os.makedirs(output_folder, exist_ok=True)

    for f in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, f))

    grouped_frames = defaultdict(list)
    filename_pattern = re.compile(r'vid(\d+)_frame(\d+)\.png') #Pattern matching how the test frames are named.

    for filename in os.listdir(input_folder):
        match = filename_pattern.match(filename)

        if match:
            video_id, frame_id = int(match.group(1)), int(match.group(2))
            grouped_frames[video_id].append((frame_id, os.path.join(input_folder, filename)))

    print(f"Found frames for {len(grouped_frames)} videos. Creating grids...")

    for video_id, frames in sorted(grouped_frames.items()):
        frames.sort()
        frame_paths = [path for _, path in frames]
        
        image_tensors = []
        for path in frame_paths:
            #Load image with OpenCV.
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            #Convert to a PyTorch Tensor and re-order dimensions.
            tensor = torch.from_numpy(img).permute(2, 0, 1)
            image_tensors.append(tensor)

        #Create the grid.
        grid_tensor = torchvision.utils.make_grid(image_tensors, nrow=len(image_tensors), padding=margin)

        #Convert the grid tensor back to an image.
        grid_image_np = grid_tensor.permute(1, 2, 0).numpy()
        grid_image_bgr = cv2.cvtColor(grid_image_np, cv2.COLOR_RGB2BGR)
        
        output_filename = f"video_{video_id}_grid.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, grid_image_bgr)

if __name__ == '__main__':
    source_folders = [
        "../natural_history_museum/test_frames",
        "../frameless/test_frames"
    ]

    output_subfolders = ["nat_hist", "frameless"]

    for folder, subfolder in zip(source_folders, output_subfolders):
        if os.path.exists(folder):
            output_folder = os.path.join("video_grids", subfolder)
            create_video_frame_grids(input_folder=folder, output_folder=output_folder)