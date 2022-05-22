import gc
from xml.sax.handler import feature_string_interning
import caer
import canaro
import os
import cv2 as cv
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
import numpy as np
import scipy


img_size = (80, 80)
channels = 1        #grey scale
char_path = r'dataset\training'


char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

# sort
char_dict = caer.sort_dict(char_dict, descending=True)
print(char_dict)

mech = []
cnt = 0
for i in char_dict:
    mech.append(i[0])
    cnt += 1
    if cnt >= 4:
        break
print(mech)

# Creating the ttraining data
train = caer.preprocess_from_dir(char_path, mech, channels=channels, IMG_SIZE=img_size, isShuffle=True)
#plt.figure(figsize=(30,30))
#plt.imshow(train[0][0], cmap='gray')
#plt.show()

featureSet, labels = caer.sep_train(train, IMG_SIZE=img_size)

#normalize the featureSet ==> (0,1)
featureSet = caer.normalize(featureSet)
labels = np_utils.to_categorical(labels, len(mech))

x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2)

# freeing space
#del train, featureSet, labels
print(gc.collect())


Batch_size = 32
Epochs = 10

#image data generator

datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size = Batch_size)

#creating the model
model = canaro.models.createSimpsonsModel(IMG_SIZE=img_size, channels=channels, output_dim=len(mech), loss='binary_crossentropy', decay=1e-6, learning_rate= 0.001, momentum=0.9, nesterov=True)

#model.summary()

callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]
training = model.fit(train_gen, steps_per_epoch = len(x_train)//Batch_size, epochs = Epochs, validation_data = (x_val, y_val), validation_steps = len(y_val)//Batch_size, callbacks = callbacks_list)

test_path = r'dataset\testing\bolt\CDQM20_CQ_M5x35L_20.png'
img = cv.imread(test_path)


def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, img_size)
    img = caer.reshape(img, img_size, channels=channels)
    return img

plt.imshow(img)
plt.show()

predictions = model.predict(prepare(img))

print(mech[np.argmax(predictions[0])])