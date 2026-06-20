import os
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, confusion_matrix, matthews_corrcoef, jaccard_score
from util import *

from keras.callbacks import  ModelCheckpoint
import scipy.misc as mc

data_location = ''
testing_images_loc = data_location + 'Chase/test/image/'
testing_label_loc = data_location + 'Chase/test/label/'

test_files = os.listdir(testing_images_loc)
test_data = []
test_label = []
desired_size=1008
for i in test_files:
    im = mc.imread(testing_images_loc + i)
    label = mc.imread(testing_label_loc + "Image_" + i.split('_')[1].split(".")[0] + "_1stHO.png")
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
    temp = cv2.resize(new_label,
                      (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    test_label.append(temp)
test_data = np.array(test_data)
test_label = np.array(test_label)

x_test = test_data.astype('float32') / 255.

y_test = test_label.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_test = np.reshape(y_test, (len(y_test), desired_size, desired_size, 1))  # adapt this if using `channels_first` im
y_test=crop_to_shape(y_test,(len(y_test), 960, 999, 1))

from  RSAN import *
model=RSANet(input_size=(desired_size,desired_size,3),start_neurons=16,keep_prob=0.78,lr=1e-3)
weight="Chase/Model/RSAN_bce.h5"

if os.path.isfile(weight): model.load_weights(weight)

model_checkpoint = ModelCheckpoint(weight, monitor='val_acc', verbose=1, save_best_only=True)
y_pred = model.predict(x_test)
y_pred= crop_to_shape(y_pred,(8,960,999,1))

saved_image_dir = './Chase/test/result_bce'
os.makedirs(saved_image_dir, exist_ok=True)

y_pred_threshold = []
i=0
for y in y_pred:
    _, temp = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
    y_pred_threshold.append(temp)
    y = y * 255

    saved_image_name = '%d.png' % i
    saved_image_path = os.path.join(saved_image_dir, saved_image_name)
    cv2.imwrite(saved_image_path, y)
    i+=1
y_test = list(np.ravel(y_test))
y_pred_threshold = list(np.ravel(y_pred_threshold))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

# Original metrics
print('Sensitivity:', recall_score(y_test, y_pred_threshold))
print('Specificity:', tn / (tn + fp))
print("F1:", 2*tp/(2*tp+fn+fp))
print('Accuracy:', accuracy_score(y_test, y_pred_threshold))
print('AUC:', roc_auc_score(y_test, list(np.ravel(y_pred))))

# Additional metrics
def dice_score(actual, predicted):
    actual = np.asarray(actual).astype(np.bool)
    predicted = np.asarray(predicted).astype(np.bool)
    im_sum = actual.sum() + predicted.sum()
    if im_sum == 0: return 1
    intersection = np.logical_and(actual, predicted)
    return 2. * intersection.sum() / im_sum

print('Dice Coefficient:', dice_score(y_test, y_pred_threshold))
print('MCC:', matthews_corrcoef(y_test, y_pred_threshold))
print('Jaccard Index:', jaccard_score(y_test, y_pred_threshold))


