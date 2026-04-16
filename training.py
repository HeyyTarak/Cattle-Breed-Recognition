import tensorflow as tf
from tensorflow.keras import layers, models
import os
import shutil
from PIL import Image

# 1. Prepare Dataset
dataset_zip_path = 'dataset.zip'
extracted_dir = 'dataset'

if not os.path.exists(dataset_zip_path):
    print(f"Error: {dataset_zip_path} not found.")
    exit()

if os.path.exists(extracted_dir):
    shutil.rmtree(extracted_dir)

print(f"Extracting {dataset_zip_path}...")
shutil.unpack_archive(dataset_zip_path, '.', 'zip')
print("Extraction complete.")

# 2. Fix corrupted images
print("Checking dataset for corrupted images...")
for root, _, files in os.walk(extracted_dir):
    for file in files:
        if file.startswith('.'):
            continue
        path = os.path.join(root, file)
        try:
            with Image.open(path) as img:
                img.verify()
        except:
            print(f"Removing corrupted: {path}")
            os.remove(path)

print("Dataset cleaned.")

# 3. Load Dataset
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.utils.image_dataset_from_directory(
    extracted_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    extracted_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Normalize
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4. Build Model (Improved)
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Dropout(0.3),   # 🔥 reduces overconfidence

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. Train
print("Training started...")
model.fit(train_ds, validation_data=val_ds, epochs=10)

# 6. Save
model.save("breed_classifier.h5")
print("Model saved successfully!")