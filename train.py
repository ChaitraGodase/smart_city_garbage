import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ===============================
# 1. Dataset paths
# ===============================
dataset_path = "garbage_classification"  # <-- your dataset folder containing 12 class folders
output_path = "data"      # <-- this will contain train/ and val/

train_dir = os.path.join(output_path, "train")
val_dir = os.path.join(output_path, "val")

# ===============================
# 2. Split dataset automatically (80% train, 20% val)
# ===============================
def split_dataset(dataset_path, train_dir, val_dir, train_ratio=0.8):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create class folders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Copy images
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

    print("✅ Dataset split completed. Train & Val folders created!")

# Split only if not already split
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    split_dataset(dataset_path, train_dir, val_dir)
else:
    print("⚠️ Train/Val folders already exist. Skipping split.")

# ===============================
# 3. Image preprocessing
# ===============================
img_height, img_width = 128, 128
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# ===============================
# 4. Build CNN Model
# ===============================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ===============================
# 5. Train Model
# ===============================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

import json
# after creating train_generator
class_indices = train_generator.class_indices
with open('labels.json', 'w') as f:
    json.dump(class_indices, f)
print("Saved class index mapping to labels.json:", class_indices)


# ===============================
# 6. Save Model
# ===============================
model.save("garbage_model.h5")
print("✅ Model saved as garbage_model.h5")
