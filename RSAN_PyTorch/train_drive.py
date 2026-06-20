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
parser = argparse.ArgumentParser(description="Train RSANet on DRIVE dataset")
parser.add_argument('--desired_size', type=int, default=592, help='Desired image size (default: 592)')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for DataLoader (default: 2)')
parser.add_argument('--weight_name', type=str, default='RSAN.pth', help='Name of weights file (default: RSAN.pth)')
parser.add_argument('--restore', action='store_true', help='Restore weights from file if exists')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
parser.add_argument('--patience', type=int, default=100, help='Early stopping patience (default: 100)')
parser.add_argument('--loss_name', type=str, default='bce', choices=['bce', 'soft_dice_cldice'], help='Loss function name (default: bce)')
parser.add_argument('--exclude_background', action='store_true', help='Exclude background channel from loss')
args = parser.parse_args()

# Set visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dynamic data location search
data_location = ''
if not os.path.exists(data_location + 'Drive'):
    data_location = '../'

training_images_loc = data_location + 'Drive/train/images/'
training_label_loc = data_location + 'Drive/train/labels/'

validate_images_loc = data_location + 'Drive/validate/images/'
validate_label_loc = data_location + 'Drive/validate/labels/'

train_files = os.listdir(training_images_loc)
train_data = []
train_label = []
validate_files = os.listdir(validate_images_loc)
validate_data = []
validate_label = []

desired_size = args.desired_size
for i in train_files:
    im = imageio.imread(training_images_loc + i)
    label = imageio.imread(training_label_loc + i.split('_')[0] + '_manual1.png', pilmode="L")
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
    label = imageio.imread(validate_label_loc + i.split('_')[0] + '_manual1.png', pilmode="L")
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

# Load into Tensor Dataset
train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(x_validate), torch.tensor(y_validate))

from RSAN import *
model = RSANet(input_size=(desired_size, desired_size, 3), start_neurons=16, keep_prob=0.85, lr=args.lr)
model = model.to(device)

weight_default = 'RSAN.pth'
if args.weight_name == weight_default:
    snapshot_path = os.path.join(data_location, "Drive/Model")
else:
    snapshot_path = os.path.join(data_location, "Drive", args.weight_name)

weight = os.path.join(snapshot_path, args.weight_name)
os.makedirs(snapshot_path, exist_ok=True)

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
from trainer import RSANTrainer

writer = SummaryWriter(log_dir=os.path.join(snapshot_path, "log"))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.loss_name == 'bce':
    criterion = nn.BCELoss()
elif args.loss_name == 'soft_dice_cldice':
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../clDice')))
    from cldice_loss.pytorch.cldice import soft_dice_cldice
    
    class SoftDiceclDiceLossWrapper(nn.Module):
        def __init__(self, iter_=3, alpha=0.5, smooth=1.0, exclude_background=False):
            super().__init__()
            self.exclude_background = exclude_background
            self.loss_fn = soft_dice_cldice(iter_=iter_, alpha=alpha, smooth=smooth, exclude_background=exclude_background)
            
        def forward(self, outputs, targets):
            # Save original exclude_background setting
            orig_exclude = self.loss_fn.exclude_background
            # If outputs channel dimension is 1, we must override exclude_background to False
            if outputs.shape[1] == 1:
                self.loss_fn.exclude_background = False
            try:
                loss = self.loss_fn(targets, outputs)
            finally:
                # Restore original exclude_background setting
                self.loss_fn.exclude_background = orig_exclude
            return loss
            
    criterion = SoftDiceclDiceLossWrapper(iter_=3, alpha=0.5, smooth=1.0, exclude_background=args.exclude_background)

trainer = RSANTrainer(
    model=model,
    dataset=train_dataset,
    optimizer=optimizer,
    criterion=criterion,
    writer=writer,
    val_dataset=val_dataset,
    patience=args.patience,  # Early stopping patience
    early_stopping_mode='max',
    early_stopping_metric='val_acc',
    snapshot_path=snapshot_path,
    batch_size=args.batch_size,
    device=device
)

# Handle resume logic: prioritize latest_model.pth if it exists, otherwise fall back to weight (RSAN.pth)
resume_path = None
if args.restore:
    latest_path = os.path.join(snapshot_path, 'latest_model.pth')
    if os.path.exists(latest_path):
        resume_path = latest_path
    elif os.path.exists(weight):
        # Backward compatibility with original weight format (only weights)
        resume_path = weight
        # Load directly because base resume expects a full checkpoint dict
        try:
            model.load_state_dict(torch.load(weight, map_location=device))
            print(f"Restored weights from {weight}")
        except Exception as e:
            print(f"Failed to load weights directly from {weight}: {e}")

print("Starting training...")
trainer.train(max_epochs=args.epochs, resume_path=resume_path)
writer.close()

