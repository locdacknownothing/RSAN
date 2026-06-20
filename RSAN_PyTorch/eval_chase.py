import os
import numpy as np
import cv2
import imageio
import torch
import json
import time
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, confusion_matrix
from util import *
import argparse

# Set visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate RSANet on CHASE DB1 dataset")
parser.add_argument('--desired_size', type=int, default=1008, help='Desired image size (default: 1008)')
parser.add_argument('--weight_name', type=str, default='RSAN.pth', help='Name of weights file (default: RSAN.pth)')
args = parser.parse_args()

# Dynamic data location search
data_location = ''
if not os.path.exists(data_location + 'Chase'):
    data_location = '../'

testing_images_loc = data_location + 'Chase/test/image/'
testing_label_loc = data_location + 'Chase/test/label/'

test_files = os.listdir(testing_images_loc)
test_data = []
test_label = []
desired_size = args.desired_size

for i in test_files:
    im = imageio.imread(testing_images_loc + i)
    label = imageio.imread(testing_label_loc + "Image_" + i.split('_')[1].split(".")[0] + "_1stHO.png", pilmode="L")
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
    
    test_data.append(cv2.resize(new_im, (desired_size, desired_size)))
    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    test_label.append(temp)

test_data = np.array(test_data)
test_label = np.array(test_label)

x_test = test_data.astype('float32') / 255.
y_test = test_label.astype('float32') / 255.

x_test = np.reshape(x_test, (len(x_test), desired_size, desired_size, 3))
y_test = np.reshape(y_test, (len(y_test), desired_size, desired_size, 1))

# Crop the ground truth to original dimensions for metric calculation
y_test = crop_to_shape(y_test, (len(y_test), 960, 999, 1))

from RSAN import *
model = RSANet(input_size=(desired_size, desired_size, 3), start_neurons=16, keep_prob=0.78, lr=1e-3)
model = model.to(device)

weight_default = 'RSAN.pth'
if args.weight_name == weight_default:
    snapshot_path = os.path.join(data_location, "Chase/Model")
else:
    snapshot_path = os.path.join(data_location, "Chase", args.weight_name)

weight = os.path.join(snapshot_path, "best_model.pth")
# weight = os.path.join(snapshot_path, "latest_model.pth")
if not os.path.exists(weight):
    weight = os.path.join(snapshot_path, args.weight_name)

if os.path.isfile(weight):
    checkpoint = torch.load(weight, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded weights from {weight}")
else:
    print(f"Warning: weights not found at {weight}")

result_dir = os.path.join(data_location, 'Chase/test/result')
os.makedirs(result_dir, exist_ok=True)

model.eval()

# Transpose x_test to PyTorch format [B, C, H, W]
x_test_py = np.transpose(x_test, (0, 3, 1, 2))
x_test_tensor = torch.tensor(x_test_py, dtype=torch.float32).to(device)

with torch.no_grad():
    y_preds = []
    # Predict in small batches of 1 to avoid memory issues (since size is 1008x1008)
    for start_idx in range(0, len(x_test)):
        batch_x = x_test_tensor[start_idx : start_idx + 1]
        batch_y_pred = model(batch_x)
        y_preds.append(batch_y_pred.cpu().numpy())
    y_pred = np.concatenate(y_preds, axis=0)

# Transpose predictions back to channels_last [B, H, W, C] for saving and crop_to_shape
y_pred = np.transpose(y_pred, (0, 2, 3, 1))

y_pred = crop_to_shape(y_pred, (len(x_test), 960, 999, 1))

y_pred_threshold = []
i = 0
for y in y_pred:
    _, temp = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
    y_pred_threshold.append(temp)
    y_img = y * 255
    file_name = '%d.png' % i
    file_path = os.path.join(result_dir, file_name)
    cv2.imwrite(file_path, y_img)
    i += 1

y_test_ravel = list(np.ravel(y_test))
y_pred_threshold_ravel = list(np.ravel(y_pred_threshold))

tn, fp, fn, tp = confusion_matrix(y_test_ravel, y_pred_threshold_ravel).ravel()

sensitivity = recall_score(y_test_ravel, y_pred_threshold_ravel)
specificity = tn / (tn + fp)
f1 = 2 * tp / (2 * tp + fn + fp)
accuracy = accuracy_score(y_test_ravel, y_pred_threshold_ravel)
auc = roc_auc_score(y_test_ravel, list(np.ravel(y_pred)))

import medpy.metric.binary as mmb

dice_list = []
hd95_list = []

for idx in range(len(y_test)):
    gt_slice = np.squeeze(y_test[idx])
    pred_slice = np.squeeze(y_pred_threshold[idx])
    
    gt_slice = (gt_slice > 0.5).astype(np.uint8)
    pred_slice = (pred_slice > 0.5).astype(np.uint8)
    
    if pred_slice.sum() > 0 and gt_slice.sum() > 0:
        dice_val = mmb.dc(pred_slice, gt_slice)
        hd95_val = mmb.hd95(pred_slice, gt_slice)
    elif pred_slice.sum() > 0 and gt_slice.sum() == 0:
        dice_val = 1.0
        hd95_val = 0.0
    else:
        dice_val = 0.0
        hd95_val = 0.0
        
    dice_list.append(dice_val)
    hd95_list.append(hd95_val)

mean_dice = np.mean(dice_list)
mean_hd95 = np.mean(hd95_list)

metric_dict = {
    "Sensitivity": sensitivity,
    "Specificity": specificity,
    "F1": f1,
    "Accuracy": accuracy,
    "AUC": auc,
    "Dice": mean_dice,
    "HD95": mean_hd95,
}

weight_prefix = os.path.splitext(args.weight_name)[0]
result_file_path = os.path.join(result_dir, f"../metric_{weight_prefix}.json")
with open(result_file_path, "w") as f:
    json.dump(metric_dict, f, indent=4)

print("Evaluation metrics saved to:", os.path.abspath(result_file_path))
print(json.dumps(metric_dict, indent=4))
