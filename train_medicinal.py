"""
Training script for Medicinal Plant Recognition Model
Uses EfficientNetB3 with transfer learning on the IMFI Indian Medicinal Flower Dataset (28 classes)
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split


def load_dataset(data_dir):
    """Walk through subdirectories and build a DataFrame of file paths + labels."""
    filepaths = []
    labels = []
    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if os.path.isfile(fpath):
                filepaths.append(fpath)
                labels.append(folder)
    df = pd.DataFrame({"filepaths": filepaths, "labels": labels})
    print(f"Found {len(df)} images across {df['labels'].nunique()} classes")
    print(f"Classes: {sorted(df['labels'].unique())}")
    return df


def build_model(num_classes, img_shape=(224, 224, 3)):
    """Build EfficientNetB3 model with custom classification head."""
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=img_shape)

    # Freeze base model initially
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


if __name__ == "__main__":
    # ─── Configuration ───────────────────────────────────────────────
    DATA_DIR = r"C:\Users\KIIT0001\Downloads\Mini Project\IMFI Indian Medicinal Flower Image Dataset\IMFI Indian Medicinal Flower Image Dataset\IMFI Dataset"
    MODEL_SAVE_PATH = "medicinal_model.keras"
    CLASS_INDICES_PATH = "medicinal_class_indices.json"

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    INITIAL_EPOCHS = 30      # Train with frozen base
    FINE_TUNE_EPOCHS = 20    # Fine-tune with unfrozen base
    LEARNING_RATE = 0.001
    FINE_TUNE_LR = 0.0001
    # ──────────────────────────────────────────────────────────────────

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Dataset directory not found at:\n  {DATA_DIR}")
        print("Please ensure the IMFI dataset is extracted correctly.")
        exit(1)

    # Load and split dataset
    print("\n[1/5] Loading dataset...")
    df = load_dataset(DATA_DIR)

    train_df, temp_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=42, stratify=df['labels'])
    valid_df, test_df = train_test_split(temp_df, train_size=0.5, shuffle=True, random_state=42, stratify=temp_df['labels'])
    print(f"Split: {len(train_df)} train / {len(valid_df)} validation / {len(test_df)} test")

    # Data generators with augmentation
    print("\n[2/5] Creating data generators...")
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='filepaths', y_col='labels',
        target_size=IMG_SIZE, class_mode='categorical',
        batch_size=BATCH_SIZE, shuffle=True
    )
    valid_gen = val_datagen.flow_from_dataframe(
        valid_df, x_col='filepaths', y_col='labels',
        target_size=IMG_SIZE, class_mode='categorical',
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_gen = val_datagen.flow_from_dataframe(
        test_df, x_col='filepaths', y_col='labels',
        target_size=IMG_SIZE, class_mode='categorical',
        batch_size=BATCH_SIZE, shuffle=False
    )

    # Save class indices
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"Saved class indices to {CLASS_INDICES_PATH}")

    num_classes = len(train_gen.class_indices)
    print(f"Number of classes: {num_classes}")

    # Build model
    print("\n[3/5] Building EfficientNetB3 model...")
    model, base_model = build_model(num_classes)
    model.compile(
        optimizer=Adamax(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    # Phase 1: Train with frozen base (transfer learning)
    print(f"\n[4/5] Phase 1: Training classification head ({INITIAL_EPOCHS} epochs)...")
    history1 = model.fit(
        train_gen,
        epochs=INITIAL_EPOCHS,
        validation_data=valid_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Fine-tune — unfreeze top layers of base model
    print(f"\n[5/5] Phase 2: Fine-tuning ({FINE_TUNE_EPOCHS} epochs)...")
    base_model.trainable = True
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adamax(learning_rate=FINE_TUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_gen,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=valid_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    print("\n" + "="*50)
    print("EVALUATION ON TEST SET")
    print("="*50)
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Loss: {test_loss:.4f}")

    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Class indices saved to: {CLASS_INDICES_PATH}")
    print("\nDone! You can now run: python test_model.py")
