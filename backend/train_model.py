# ==========================================
# AI SKIN DISEASE DETECTION - MODEL TRAINING
# ==========================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ==========================================
# DATASET PATH
# ==========================================

dataset_path = "../dataset"

train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

# ==========================================
# IMAGE DATA GENERATOR
# ==========================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

# ==========================================
# CNN MODEL
# ==========================================

model = models.Sequential([

    layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128,activation='relu'),

    layers.Dense(train_generator.num_classes,activation='softmax')

])

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("../model", exist_ok=True)

model.save("../model/skin_model.h5")

print("\nModel trained and saved successfully.")
