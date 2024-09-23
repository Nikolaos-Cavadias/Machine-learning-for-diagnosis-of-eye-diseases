

###         Step 1: Define the FR-UNet Model

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class FRUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FRUNet, self).__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.center = ConvBlock(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        center = self.center(self.pool(enc4))
        
        dec4 = self.up4(center)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.up3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)
        
        final = self.final(dec1)
        return final









###         Step 2: Prepare the Data Pipeline


import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np

class RetinalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)
        self.mask_list = os.listdir(mask_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Load as grayscale

        # Convert mask to binary
        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)
        mask = Image.fromarray(mask)  # Convert numpy array back to PIL Image

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        mask = mask.squeeze(0)  # Ensure mask shape is [H, W]
        
        return image, mask

# Define transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Create dataset 
##dataset = RetinalDataset(r"C:\Users\Nikolas\Downloads\DRIVE\training\images", r"C:\Users\Nikolas\Downloads\DRIVE\training\1st_manual", transform=transform)
##dataset = RetinalDataset(r"C:\Users\Nikolas\Downloads\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation ORIGINAL\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation\train\A", r"C:\Users\Nikolas\Downloads\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation ORIGINAL\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation\train\gd A", transform=transform)
dataset = RetinalDataset(r"/mnt/iusers01/fse-ugpgt01/mace01/x93189nc/UNET/Raw NADG", r"/mnt/iusers01/fse-ugpgt01/mace01/x93189nc/UNET/gd NADG", transform=transform)


# Train-test split (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)






###         Step 3: Train the Model
import torch.optim as optim
import torch.nn.functional as F

# Hyperparameters
num_epochs = 20
learning_rate = 0.001

# Initialize model, loss function, and optimizer
model = FRUNet(in_channels=3, out_channels=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.float()
            masks = masks.float().unsqueeze(1)  # Ensure masks have the same channel dimension as outputs
            outputs = model(images)
            predicted = torch.sigmoid(outputs) > 0.5
            total += masks.numel()
            correct += (predicted == masks).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, masks in train_loader:
        images = images.float()
        masks = masks.float().unsqueeze(1)  # Ensure masks have the same channel dimension as outputs

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Calculate accuracy
    train_accuracy = calculate_accuracy(train_loader, model)
    test_accuracy = calculate_accuracy(test_loader, model)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, "
          f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")







###     Step 4: Inference
import os
import matplotlib.pyplot as plt
from PIL import Image

def predict_and_save(model, image_dir, transform, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            output = model(image)
            prediction = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
            prediction = (prediction > 0.5).astype(np.uint8)  # Binarize output
        
        # Convert numpy array to PIL Image and save
        prediction_image = Image.fromarray(prediction * 255)  # Convert binary to grayscale
        output_path = os.path.join(output_dir, image_name)
        prediction_image.save(output_path)
        
# Load a pre-trained model
# model.load_state_dict(torch.load('path/to/saved_model.pth'))

# Predict on a new image
image_dir = r"/mnt/iusers01/fse-ugpgt01/mace01/x93189nc/UNET/ALL DATA original/"
output_dir = r"/mnt/iusers01/fse-ugpgt01/mace01/x93189nc/UNET/NEW GROUND TRUTHS/"
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
predict_and_save(model, image_dir, transform, output_dir)


