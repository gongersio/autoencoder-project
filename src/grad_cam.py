import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F

from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from models import FinalEncoder
from train_model import load_dataset
from load_dataset import FrameDataset

class GradCAM:
    """A class to generate Grad-CAM heatmaps by capturing gradients and activations from a target layer."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks to the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def _remove_hooks(self):
        """Remove the hooks to clean up."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate_heatmap(self, input_tensor, class_idx=None):
        """Generates a heatmap for a given input tensor and class index."""
        self.model.eval()
        
        #Forward pass to get logits.
        logits, _ = self.model(input_tensor)

        #If class_idx is not provided, use the predicted class.
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        #Get the score for the target class.
        target_score = logits[0, class_idx]
        
        self.model.zero_grad()

        #Backward pass to compute gradients.
        target_score.backward(retain_graph=True)

        #Pool gradients across spatial dimensions to get channel importance weights.
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        #Weight feature maps by the corresponding gradient-based importance.
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        #Average the weighted feature maps along the channel dimension
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        #Apply ReLU to keep only positive influences.
        heatmap = F.relu(heatmap)
        
        #Normalize the heatmap to be between 0 and 1
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()
    
    def __del__(self):
        """Ensure hooks are removed when the object is deleted."""
        self._remove_hooks()

def show_cam_on_image(img, heatmap, output_path, threshold=0.05, alpha=0.4):
    """Overlays the heatmap on the original image where activations are  above a threshold and saves the result."""
    # Resize the single-channel heatmap to match the image dimensions
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Create a mask where the heatmap intensity is above the threshold
    mask = resized_heatmap > threshold

    # Scale the heatmap to 0-255 and apply the colormap
    colored_heatmap = np.uint8(255 * resized_heatmap)
    colored_heatmap = cv2.applyColorMap(colored_heatmap, cv2.COLORMAP_JET)

    # Create a copy of the original image to draw on
    superimposed_img = img.copy()

    # Apply the colored heatmap only on the regions defined by the mask
    # This uses a standard alpha blending formula
    superimposed_img[mask] = (colored_heatmap[mask] * alpha) + (superimposed_img[mask] * (1 - alpha))
    
    # Save the final image
    cv2.imwrite(output_path, superimposed_img)

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "models/model_83_NEW.pth"
    TEST_DATASET_PATH = "datasets/test_model.pkl"
    OUTPUT_DIR = "gradcam"
    NUM_IMAGES = 40

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #Clear existing files in the output directory.
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

    model = FinalEncoder(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Loading the dataset...")
    test_dataset = load_dataset(TEST_DATASET_PATH)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    #The 'last_conv_layer' method in FinalEncoder is used to get the target layer.
    grad_cam = GradCAM(model=model, target_layer=model.last_conv_layer())

    print(f"Generating Grad-CAM for {NUM_IMAGES} images...")
    
    processed_images = 0
    for i, (image_tensor, true_label, video_id) in enumerate(test_loader):
        if processed_images >= NUM_IMAGES:
            break

        image_tensor = image_tensor.to(DEVICE)
        
        #Get model prediction.
        logits, _ = model(image_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()

        #Generate the heatmap for the predicted class.
        heatmap = grad_cam.generate_heatmap(image_tensor, class_idx=predicted_class)
        
        #Convert tensor back to a displayable format (PIL -> NumPy -> BGR).
        pil_img = to_pil_image(image_tensor.squeeze().cpu())
        original_img_rgb = np.array(pil_img)
        original_img_bgr = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)

        filename = f"frame{i}_pred{predicted_class}_true{true_label.item()}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        show_cam_on_image(original_img_bgr, heatmap, output_path)
        
        processed_images += 1
        
    print("\n--- Grad-CAM Generation Complete ---")