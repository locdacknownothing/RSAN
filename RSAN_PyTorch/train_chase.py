import os
import numpy as np
import cv2
import imageio
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse

# Set seed
np.random.seed(42)
torch.manual_seed(42)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train RSANet on CHASE DB1 dataset")
parser.add_argument('--desired_size', type=int, default=1008, help='Desired image size (default: 1008)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader (default: 1)')
parser.add_argument('--weight_name', type=str, default='RSAN.pth', help='Name of weights file (default: RSAN.pth)')
parser.add_argument('--restore', action='store_true', default=True, help='Restore weights from file if exists (default: True)')
parser.add_argument('--no_restore', dest='restore', action='store_false', help='Do not restore weights from file')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
args = parser.parse_args()

# Set visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dynamic data location search
data_location = ''
if not os.path.exists(data_location + 'Chase'):
    data_location = '../'

training_images_loc = data_location + 'Chase/train/image/'
training_label_loc = data_location + 'Chase/train/label/'
validate_images_loc = data_location + 'Chase/validate/images/'
validate_label_loc = data_location + 'Chase/validate/labels/'

train_files = os.listdir(training_images_loc)
train_data = []
train_label = []
validate_files = os.listdir(validate_images_loc)
validate_data = []
validate_label = []

desired_size = args.desired_size
for i in train_files:
    im = imageio.imread(training_images_loc + i)
    label = imageio.imread(training_label_loc + "Image_" + i.split('_')[1].split(".")[0] + "_1stHO.png", pilmode="L")
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    train_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    train_label.append(temp)

for i in validate_files:
    im = imageio.imread(validate_images_loc + i)
    label = imageio.imread(validate_label_loc + "Image_" + i.split('_')[1].split(".")[0] + "_1stHO.png", pilmode="L")
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    validate_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    validate_label.append(temp)

train_data = np.array(train_data)
train_label = np.array(train_label)

validate_data = np.array(validate_data)
validate_label = np.array(validate_label)

x_train = train_data.astype('float32') / 255.
y_train = train_label.astype('float32') / 255.

x_validate = validate_data.astype('float32') / 255.
y_validate = validate_label.astype('float32') / 255.

# Reshape & transpose to PyTorch format [B, C, H, W]
x_train = np.reshape(x_train, (len(x_train), desired_size, desired_size, 3))
y_train = np.reshape(y_train, (len(y_train), desired_size, desired_size, 1))
x_validate = np.reshape(x_validate, (len(x_validate), desired_size, desired_size, 3))
y_validate = np.reshape(y_validate, (len(y_validate), desired_size, desired_size, 1))

x_train = np.transpose(x_train, (0, 3, 1, 2))
y_train = np.transpose(y_train, (0, 3, 1, 2))
x_validate = np.transpose(x_validate, (0, 3, 1, 2))
y_validate = np.transpose(y_validate, (0, 3, 1, 2))

# Load into Tensor Dataset and DataLoaders
train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(x_validate), torch.tensor(y_validate))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

from RSAN import *
model = RSANet(input_size=(desired_size, desired_size, 3), start_neurons=16, keep_prob=0.78, lr=1e-3)
model = model.to(device)

weight = os.path.join(data_location, "Chase/Model", args.weight_name)
os.makedirs(os.path.dirname(weight), exist_ok=True)

if args.restore and os.path.isfile(weight):
    model.load_state_dict(torch.load(weight, map_location=device))
    print(f"Restored weights from {weight}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

print("Starting training...")
epochs = args.epochs
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        preds = (outputs > 0.5).float()
        train_acc += (preds == targets).float().mean().item() * inputs.size(0)
        
    train_loss /= len(train_dataset)
    train_acc /= len(train_dataset)
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            val_acc += (preds == targets).float().mean().item() * inputs.size(0)
            
    val_loss /= len(val_dataset)
    val_acc /= len(val_dataset)
    
    print(f"Epoch {epoch}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
    
    # Save checkpoint
    torch.save(model.state_dict(), weight)
