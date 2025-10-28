from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2  # OpenCV for pre-processing
import numpy as np
import random

# Initialize Flask app
app = Flask(__name__, static_url_path='', static_folder='.')

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the model
num_classes = 7
num_stages = 3
class_map = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
stage_map = {'stage_0': 0, 'stage_1': 1, 'stage_2': 2}

category_map = {v: k for k, v in class_map.items()}
stage_map = {v: k for k, v in stage_map.items()}

class SkinCancerCNN(nn.Module):
    def __init__(self):
        super(SkinCancerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)  # For classification
        self.fc4 = nn.Linear(256, num_stages)   # For stage detection
        self.dropout = nn.Dropout(0.5)          # Dropout layer with 50% drop probability

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        category_output = self.fc3(x)
        stage_output = self.fc4(x)
        return category_output, stage_output

model = SkinCancerCNN()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def is_skin_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
    skin_percentage = (mask > 0).mean()
    return skin_percentage > 0.5

def predict(image_path, model, transform, threshold=0.5, scaling_factor=1.0):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs_category, outputs_stage = model(image)
        
        # Scale logits before applying softmax
        scaled_outputs_category = outputs_category * scaling_factor
        scaled_outputs_stage = outputs_stage * scaling_factor
        
        softmax_category = torch.nn.functional.softmax(scaled_outputs_category, dim=1)
        softmax_stage = torch.nn.functional.softmax(scaled_outputs_stage, dim=1)
        
        max_confidence_category, predicted_category = torch.max(softmax_category.data, 1)
        max_confidence_stage, predicted_stage = torch.max(softmax_stage.data, 1)

        if max_confidence_category.item() < threshold:
            return "Not a cancer", "Not a cancer"

        random_category = random.choice(list(class_map.keys()))

    return random_category, stage_map[predicted_stage.item()]

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        if not is_skin_image(filepath):
            return jsonify({'category': "Not a skin cancer", 'stage': "Not a skin cancer"})

        predicted_category, predicted_stage = predict(filepath, model, transform)
        return jsonify({'category': predicted_category, 'stage': predicted_stage})

    return jsonify({'error': 'File upload failed'}), 500

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
