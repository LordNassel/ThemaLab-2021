#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path
import gc
import cv2
import random
import numpy as np

import seaborn as sns
import tensorflow as tf
import opendatasets as od
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#################################################
#####   Downloading Data with Kaggle API    #####
#################################################

print('Downloading files...')
od.download("https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri")

################################
#####   Loading Dataset    #####
################################

base_dir = "/home/gabor/Temalabor/Try2/brain-tumor-classification-mri"
train_dir = os.path.join(base_dir, "Training")
test_dir = os.path.join(base_dir, "Testing")

train_glioma_dir = os.path.join(train_dir, "glioma_tumor")
train_meningioma_dir = os.path.join(train_dir, "meningioma_tumor")
train_healthy_dir = os.path.join(train_dir, "no_tumor")
train_pituitary_dir = os.path.join(train_dir, "pituitary_tumor")

test_glioma_dir = os.path.join(test_dir, "glioma_tumor")
test_meningioma_dir = os.path.join(test_dir, "meningioma_tumor")
test_healthy_dir = os.path.join(test_dir, "no_tumor")
test_pituitary_dir = os.path.join(test_dir, "pituitary_tumor")

#####
train_glioma_fnames = os.listdir(train_glioma_dir)
train_glioma_fnames.sort()

train_meningioma_fnames = os.listdir(train_meningioma_dir)
train_meningioma_fnames.sort()

train_healthy_fnames = os.listdir(train_healthy_dir)
train_healthy_fnames.sort()

train_pituitary_fnames = os.listdir(train_pituitary_dir)
train_pituitary_fnames.sort()

######
test_glioma_fnames = os.listdir(test_glioma_dir)
test_glioma_fnames.sort()

test_meningioma_fnames = os.listdir(test_meningioma_dir)
test_meningioma_fnames.sort()

test_healthy_fnames = os.listdir(test_healthy_dir)
test_healthy_fnames.sort()

test_pituitary_fnames = os.listdir(test_pituitary_dir)
test_pituitary_fnames.sort()

######################################
#####   Data Visualization I.    #####
######################################

colors_dark = ["#1F1F1F", "#313131", "#636363", "#AEAEAE", "#DADADA"]
colors_green = ["#01411C", "#4B6F44", "#4F7942", "#74C365", "#D0F0C0"]
labels = ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"]
TrueLabels = ["True Glioma", "True Healthy", "True Meningioma", "True Pituitary"]
PredictedLabels = ["Predicted Glioma", "Predicted Healthy", "Predicted Meningioma", "Predicted Pituitary"]

nrows = 4
ncols = 4
fig = plt.gcf()
fig.set_size_inches(16, 16)

next_glioma_pix = [os.path.join(train_glioma_dir, fname) for fname in train_glioma_fnames[:4]]
next_meningioma_pix = [os.path.join(train_meningioma_dir, fname) for fname in train_meningioma_fnames[:4]]
next_healthy_pix = [os.path.join(train_healthy_dir, fname) for fname in train_healthy_fnames[:4]]
next_pituitary_pix = [os.path.join(train_pituitary_dir, fname) for fname in train_pituitary_fnames[:4]]

for i, img_path in enumerate(next_glioma_pix + next_meningioma_pix + next_healthy_pix + next_pituitary_pix):
    sp = plt.subplot(nrows, ncols, i + 1)
    if(i == 0):
        plt.ylabel('Glioma Tumor')
    if(i == 4): 
        plt.ylabel('Meningioma Tumor')
    if(i == 8):
        plt.ylabel('Healthy')
    if(i == 12):
        plt.ylabel('Pituitary Tumor')
        
    img = mpimg.imread(img_path)
    plt.imshow(img)
    
plt.show()

######################################
#####   Data Visualization II.   #####
######################################

train_labels = []
test_labels = []

for i in os.listdir(train_dir):
    for j in os.listdir(train_dir + "/" + i):
        train_labels.append(i)

for i in os.listdir(test_dir):
    for j in os.listdir(test_dir + "/" + i):
        test_labels.append(i)

plt.figure(figsize = (17, 8))
lis = ["Train", "Test"]
for i, j in enumerate([train_labels, test_labels]):
    plt.subplot(1, 2, i + 1)
    sns.countplot(x = j)
    plt.xlabel(lis[i])

plt.show()

#############################
####    Loading Data    #####
#############################

X_train = []
Y_train = []
image_size = 299

for i in labels:
    folderPath = os.path.join("/home/gabor/Temalabor/Try2/brain-tumor-classification-mri", "Training", i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        Y_train.append(i)

for i in labels:
    folderPath = os.path.join("/home/gabor/Temalabor/Try2/brain-tumor-classification-mri", "Testing", i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        Y_train.append(i)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train, Y_train = shuffle(X_train, 
                           Y_train, 
                           random_state = 42)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, 
                                                    Y_train, 
                                                    test_size = 0.1, 
                                                    random_state = 42)

Y_train_new = []
for i in Y_train:
    Y_train_new.append(labels.index(i))

Y_train = Y_train_new
Y_train = tf.keras.utils.to_categorical(Y_train)

Y_test_new = []
for i in Y_test:
    Y_test_new.append(labels.index(i))

Y_test = Y_test_new
Y_test = tf.keras.utils.to_categorical(Y_test)

########################################
#####   Creating Neural Network    #####
########################################

img_height = 299
img_width = 299

# Load pre-trained model without fully-connected layers
base_model = InceptionV3(weights = "imagenet", 
                         input_shape = (img_height, img_width, 3), 
                         include_top = False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(2048, activation = "relu")(x)
x = Dense(1024, activation = "relu")(x)
x = Dense(512, activation = "relu")(x)
out = Dense(4, activation = "softmax")(x)

###############################
#####   Creating Model    #####
###############################

model = Model(inputs = base_model.input,
              outputs = out)

# Freezing Layers
for layer in base_model.layers:
    layer.trainable = False

callbacks = [
    EarlyStopping(monitor = "val_loss",
                  patience = 30,
                  verbose = 1),
    ModelCheckpoint(
        filepath = "best_model.h5", 
        monitor = "val_loss", 
        save_best_only = True, 
        mode = "auto"),]

# compile the model
model.compile(optimizer = "Adam", 
              metrics = ["accuracy"], 
              loss = "categorical_crossentropy")

# model.summary()

#######################################
#####   Teaching Neural Network   #####
#######################################

history = model.fit(X_train,
                    Y_train,
                    validation_split = 0.1, 
                    epochs = 200, 
                    verbose = 1, 
                    batch_size = 64,
                    callbacks = callbacks)

print("Learning Finished.")

#######################################
#####   Plot Accuracy and Loss    #####
#######################################

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

plt.figure(figsize = (18, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label = "Training Accuracy")
plt.plot(epochs, val_acc, label = "Validation Accuracy")
plt.legend(loc = "lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label = "Training Loss")
plt.plot(epochs, val_loss, label = "Validation Loss")
plt.title("Training and Validation Loss")
plt.legend(loc = "upper right")

plt.show()

###############################################
#####   Heatmap of the Confusion Matrix   #####
###############################################

pred = model.predict(X_test)
pred = np.argmax(pred, axis = 1)
Y_test_new = np.argmax(Y_test, axis = 1)

print(classification_report(Y_test_new, pred))
fig, ax = plt.subplots(1, 1, figsize = (14, 7))

sns.heatmap(confusion_matrix(Y_test_new, pred),
            ax = ax,
            xticklabels = PredictedLabels,
            yticklabels = TrueLabels,
            annot = True,
            cmap = colors_green[::-1],
            alpha = 0.7,
            linewidths = 2,
            linecolor = colors_dark[3])

fig.text(s = "Heatmap of the Confusion Matrix",
         size = 18,
         fontweight = "bold",
         fontname = "monospace",
         color = colors_dark[1],
         y = 0.92,
         x = 0.28,
         alpha = 0.8)

plt.show()

##################################################
#####   Examining internal representations   #####
##################################################

# A new model is created, where the different layers of the model are connected according to the layer list above
successive_outputs = [model.layers[1].output,
                      model.layers[30].output,
                      model.layers[60].output,
                      model.layers[90].output]

visualization_model = Model(base_model.input, successive_outputs)

# Select pictures from all classes
glioma_img_files = [os.path.join(train_glioma_dir, f) for f in train_glioma_fnames]
meningioma_img_files = [os.path.join(train_meningioma_dir, f) for f in train_meningioma_fnames]
healthy_img_files = [os.path.join(train_healthy_dir, f) for f in train_healthy_fnames]
pituitary_img_files = [os.path.join(train_pituitary_dir, f) for f in train_pituitary_fnames]

img_path = random.choice(glioma_img_files + meningioma_img_files + healthy_img_files + pituitary_img_files)

img = load_img(img_path, target_size = (299, 299))  # this is a PIL image

x = img_to_array(img)  # Numpy array with shape (299, 299, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 299, 299, 3)

# Data preparation
x /= 255

# Run the prediction up to the given layer
successive_feature_maps = visualization_model.predict(x)

# Displaying representations by layer
for feature_map in successive_feature_maps:
    n_features = feature_map.shape[-1]  # feature maps, based on the number of convolution filters
    size = feature_map.shape[1]
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
        x = feature_map[0, :, :, i]
        # Feature maps need to be converted to be visible
        x -= x.mean()
        x /= x.std()
        x *= 64
        x += 128
        x = np.clip(x, 0, 255).astype("uint8")
        display_grid[:, i * size : (i + 1) * size] = x
        
    # Display the grid
    scale = 20.0 / n_features
    plt.figure(figsize = (scale * n_features, scale))
    plt.grid(False)
    plt.imshow(display_grid, aspect = "auto", cmap = "viridis")

#######################
#####   Finish    #####
#######################

gc.collect()



#           .            .                     .
#                  _        .                          .            (
#                 (_)        .       .                                     .
#  .        ____.--^.
#   .      /:  /    |                               +           .         .
#         /:  `--=--'   .                                                .
#  LG    /: __[\==`-.___          *           .
#       /__|\ _~~~~~~   ~~--..__            .             .
#       \   \|::::|-----.....___|~--.                                 .
#        \ _\_~~~~~-----:|:::______//---...___
#    .   [\  \  __  --     \       ~  \_      ~~~===------==-...____
#        [============================================================-
#        /         __/__   --  /__    --       /____....----''''~~~~      .
#  *    /  /   ==           ____....=---='''~~~~ .
#      /____....--=-''':~~~~                      .                .
#      .       ~--~         Kuat Drive Yard's Imperial-class Star Destroyer
#                     .                                   .           .
#                          .                      .             +
#        .     +              .                                       <=>
#                                               .                .      .
#   .                 *                 .                *                ` -