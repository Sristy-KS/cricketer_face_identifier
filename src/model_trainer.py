import os
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_cricket_model(dataset_dir="dataset", model_save_path="models/cricket_face_model.keras", epochs=10):
    os.makedirs("models", exist_ok=True)
    
    # Image Augmentation for small datasets
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        subset='validation'
    )

    # Save class index mapping to JSON
    classes = {v: k for k, v in train_gen.class_indices.items()}
    with open("models/classes.json", "w") as f:
        json.dump(classes, f)

    num_classes = len(classes)

    # Base Pre-trained Model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base weights

    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training model...")
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    model.save(model_save_path)
    print(f"Model saved successfully to {model_save_path}")