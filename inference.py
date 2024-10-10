import os
import torch
import torch.nn.functional as F
import cv2
from models.model2 import RoadSegmenter  # Import the model
from scriptss.config import Config
import numpy as np
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel

# Set proxy in the code
os.environ['http_proxy'] = 'http://172.22.22.1:1092'
os.environ['https_proxy'] = 'http://172.22.22.1:1092'


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained global model
model = RoadSegmenter()
model_path = os.path.join(Config.WEIGHTS_PATH, 'best_global_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Create output directory if it doesn't exist
output_dir = "OutputsKITTI"
os.makedirs(output_dir, exist_ok=True)

# Path to the images
image_dir = "datasets/KITTIRoad/testing/images"

# Get list of image files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

def infer_and_save(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_tensor = image_tensor.to(device)

    # Run inference
    with torch.no_grad():
        pred_mask = model(image_tensor)
        pred_mask = F.interpolate(pred_mask, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
        pred_mask = torch.sigmoid(pred_mask).cpu().numpy()

    # Post-process the predicted mask
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Binarize the mask

    # Save the mask to the output directory
    cv2.imwrite(output_path, pred_mask.squeeze())

# Loop through all the images and run inference
for image_file in tqdm(image_files, desc="Processing Images"):
    image_path = os.path.join(image_dir, image_file)

    # Modify the filename as per the required naming convention
    filename_base = os.path.splitext(image_file)[0].replace("_", "_road_")
    output_path = os.path.join(output_dir, f"{filename_base}.png")

    # Run inference and save the mask
    infer_and_save(image_path, output_path)

print(f"All masks have been saved to {output_dir}")

# After generating all masks, make sure to follow the birds-eye view transformation as per the evaluation server's instructions.
