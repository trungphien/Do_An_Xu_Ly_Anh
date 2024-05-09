import numpy as np 
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2
import tensorflow
import keras
from keras.metrics import Recall
from keras import Model, callbacks
from keras.layers import Input, Dense, add, Conv2D, MaxPool2D ,GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.utils import image_dataset_from_directory
from keras.layers import Rescaling
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

train_path = 'Pneumonia-Diagnosis-ResNet/chest_xray_pneumonia/chest_xray/train'
valid_path = 'Pneumonia-Diagnosis-ResNet/chest_xray_pneumonia/chest_xray/val'
test_path =  'Pneumonia-Diagnosis-ResNet/chest_xray_pneumonia/chest_xray/test'

BATCH_SIZE = 50
EPOCHS = 20
IMAGE_SIZE = (200, 200)

train_dataset = image_dataset_from_directory(train_path,
                                             seed=42,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMAGE_SIZE)

valid_dataset = image_dataset_from_directory(valid_path,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMAGE_SIZE)

test_dataset = image_dataset_from_directory(test_path,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMAGE_SIZE)


rescale = Rescaling(scale=1.0 / 255)
train_dataset = train_dataset.map(lambda image, label: (rescale(image), label))
valid_dataset  = valid_dataset.map(lambda image, label: (rescale(image), label))
test_dataset  = test_dataset.map(lambda image, label: (rescale(image), label))

cnt_imgs = 16  # we take 8 images for each class
norm_path = train_path + '/NORMAL'
pneumonia_path = train_path + '/PNEUMONIA'
norm_imgs = os.listdir(norm_path)[:cnt_imgs]
pneumonia_imgs = os.listdir(pneumonia_path)[:cnt_imgs]

counter = 0
norm_imgs_path = [norm_path + '/' + i for i in norm_imgs]
pneumonia_imgs_path = [pneumonia_path + '/' + j for j in pneumonia_imgs]
all_imgs = norm_imgs_path + pneumonia_imgs_path
random.shuffle(all_imgs)

plt.figure(figsize=(28, 10))
for img_path in all_imgs:
    plt.subplot(4, 8, counter + 1)
    img = cv2.imread(img_path)
    try:
        img = cv2.resize(img, IMAGE_SIZE)
    except Exception as e:
        print(str(e))
    label = img_path[len(train_path) + 1: img_path.rfind('/')]
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')
    counter += 1
plt.show()
def check_cnt_label(label: str) -> int:
    """A function that should determine the number of objects of this
    class in the specified directories"""
    cnt_object = 0
    paths = [train_path, valid_path, test_path]
    for path in paths:
        path += '/' + label
        cnt_object += len(os.listdir(path))
    return cnt_object

CNT_NORMAL = check_cnt_label('NORMAL')
CNT_PNEUMONIA = check_cnt_label('PNEUMONIA')

fig = go.Figure()
fig.add_trace(go.Bar(
    x=['NORMAL', 'PNEUMONIA'],
    y=[CNT_NORMAL, CNT_PNEUMONIA],
    name='Primary Product',
    marker_color='indianred',
    width=[0.4, 0.4]))

fig.update_layout(title='Classes and their number in the dataset', title_x=0.5)
fig.show()

inputs = Input(shape=(IMAGE_SIZE + (3,)))

x = Conv2D(32, (3, 3), activation='elu')(inputs)
x = Conv2D(64, (3, 3), activation='elu')(x)
block_1_output = MaxPool2D(pool_size=(3, 3))(x)

x = Conv2D(64, (3, 3), activation='elu', padding='same')(block_1_output)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='elu', padding='same')(x)
block_2_output = add([x, block_1_output])

x = Conv2D(64, (3, 3), activation='elu', padding='same')(block_2_output)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='elu', padding='same')(x)
block_3_output = add([x, block_2_output])

x = Conv2D(128, (3, 3), activation='elu')(block_3_output)
x = MaxPool2D(pool_size=(2, 2))(x)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='elu')(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs, output)

model.summary()

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=[Recall()])

CALLBACKS = [
    callbacks.EarlyStopping(monitor='loss', min_delta= 0, patience=8, verbose=1),  
    callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, min_delta=0.01, min_lr=0, patience=4, verbose=1, mode='auto')
]

history = model.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset, callbacks=CALLBACKS)
test_result = model.evaluate(test_dataset)
train_result = model.evaluate(train_dataset)

print(f'Metric (Recall) on test set: {test_result[1]}')
print(f'Metric (Recall) on train set: {train_result[1]}')