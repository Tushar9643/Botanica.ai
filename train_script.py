# This script mimics the training process in the notebook to ensure the model and indices are saved properly
# if the user cannot run the notebook cells manually.

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from sklearn.model_selection import train_test_split

def dense_block(units, dropout_rate, act='relu'):
    from tensorflow.keras.models import Sequential
    block = Sequential()
    block.add(Dense(units, activation=act))
    block.add(BatchNormalization())
    block.add(Dropout(dropout_rate))
    return block

def loading_the_data(data_dir):
    filepaths = []
    labels = []
    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        if not os.path.isdir(foldpath): continue
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    return df

if __name__ == "__main__":
    print("Starting training script...")
    
    # Path configuration
    data_dir = r"C:/Users/KIIT0001/Downloads/Mini Project/dataset/flowers"
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        exit()

    print("Loading data...")
    df = loading_the_data(data_dir)
    train_df, ts_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=42)
    valid_df, test_df = train_test_split(ts_df, train_size=0.5, shuffle=True, random_state=42)

    batch_size = 16
    img_size = (224, 224)
    
    print("Creating data generators...")
    tr_gen = ImageDataGenerator(rescale=1./255)
    ts_gen = ImageDataGenerator(rescale=1./255)

    train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

    # Save class indices immediately
    print("Saving class indices...")
    with open('class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f)
    
    print("Building model...")
    img_shape = (224, 224, 3)
    class_counts = len(train_gen.class_indices)
    
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=img_shape, pooling=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # Custom blocks (simplified from notebook for script)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    predictions = Dense(class_counts, activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adamax(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training model (this may take a while)...")
    # Reduced epochs for quick setup/testing, user can retrain in notebook
    epochs = 1 
    history = model.fit(train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, shuffle=False)
    
    print("Saving model...")
    model.save('efficientnetb3_flower_model.keras')
    print("Done! You can now run python inference.py")
