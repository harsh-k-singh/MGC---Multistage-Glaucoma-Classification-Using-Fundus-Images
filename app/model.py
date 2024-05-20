import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")



class TripletChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(TripletChannelAttentionBlock, self).__init__()
        reduced_channels = in_channels // reduction_ratio

        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([reduced_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([in_channels, 1, 1]),
            nn.Sigmoid()
        )

        self.branch2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((in_channels, 1)),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([1, in_channels, 1]),
            nn.Sigmoid()
        )

        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, in_channels)),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([1, 1, in_channels]),
            nn.Sigmoid()
        )

        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Handle input tensors with spatial dimensions of size 1
        if height == 1 and width == 1:
            branch1 = self.branch1(x)
            output = branch1
        else:
            branch1 = self.branch1(x)*self.weights[0]
            branch2 = torch.rot90(self.branch2(torch.rot90(x, dims=(2, 3)))*self.weights[1], dims=(2, 3))
            branch3 = torch.rot90(self.branch3(torch.rot90(x, dims=(1, 2)))*self.weights[2], dims=(1, 2))

            output = branch1 +  branch2 +  branch3

        return output





# Define SAB (Spatial Attention Block)
class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()
        # Define SAB components here
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm([1, 1, 1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through SAB
        # Compute spatial attention feature maps
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        att = self.conv(concat)
        ln_out = self.ln(att)
        att = self.sigmoid(ln_out)
        x = x * att
        return x

# Define custom dataset class
class GlaucomaDataset(Dataset):
    def __init__(self, data_dir, class_labels, transform=None):
        self.data_dir = data_dir
        self.class_labels = class_labels
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Iterate through each class folder and collect image paths and labels
        for class_label in class_labels:
            class_dir = os.path.join(data_dir, class_label)
            for filename in os.listdir(class_dir):
                if filename.endswith(('.jpg', '.png')):  # Ensure image formats
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(class_labels.index(class_label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Ensure RGB format
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class CustomClassifier(nn.Module):
    def __init__(self, num_classes, in_channels=1280):
        super(CustomClassifier, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Depthwise separable convolutions
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.global_pool(x).view(x.size(0), -1)
        se_weights = self.se(x).unsqueeze(-1)
        x = x.unsqueeze(-1) * se_weights  # Add unsqueeze(-1) to x
        x = x.squeeze(-1)  # Squeeze the added dimension
        x = self.fc(x)
        return x

class CA_Net(nn.Module):
    def __init__(self, num_classes):
        super(CA_Net, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=True).features
        self.TCAB = TripletChannelAttentionBlock(1280)
        self.SAB = SpatialAttentionBlock()
        self.dropout = nn.Dropout(0.2)
        self.classifier = CustomClassifier(num_classes)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Add a global average pooling layer

    def forward(self, x):
        feature_maps = self.backbone(x)
        feature_maps = self.global_pool(feature_maps)  # Apply global average pooling
        Fch = self.TCAB(feature_maps)
        Fsp = self.SAB(Fch)
        Fsp = self.dropout(Fsp)
        logits = self.classifier(Fsp)
        return logits

# Define data paths and labels (replace with your actual paths)
data_dir = '/content/drive/MyDrive/BTP'
class_labels = ['early_glaucoma', 'normal_control', 'advanced_glaucoma']

# Define transforms (adjust parameters as needed)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and split into train, test, and validation
dataset = GlaucomaDataset(data_dir, class_labels, transform)
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model, optimizer, and loss function
model = CA_Net(num_classes=len(class_labels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 1.2]).to(device))

# Training loop
num_epochs = 50
best_val_acc = 0.0
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_dataloader:
        images = images.to(device)  # Move images to device
        labels = labels.to(device)  # Move labels to device
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

    # Validation after each epoch
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_acc += (preds == labels).sum().item()
    val_loss /= len(val_dataset)
    val_losses.append(val_loss)
    val_acc /= len(val_dataset)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Save the best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

# Load the best model and evaluate on the test set
# Load the best model and evaluate on the test set
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)  # Move input images to GPU
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = (torch.tensor(predictions) == torch.tensor(true_labels)).float().mean().item()
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
conf_matrix = confusion_matrix(true_labels, predictions)
print(f"Test Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")
print("Confusion Matrix:")
print(conf_matrix)