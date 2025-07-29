import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, backend
from sklearn.model_selection import train_test_split

from data import generate_samples, preprocess
from weights_logger_callback import WeightsLogger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
local_project_path = '.'
local_data_path = os.path.join(local_project_path, 'data')

if __name__ == '__main__':
    # Load driving data
    df = pd.read_csv(os.path.join(local_data_path, 'driving_log.csv'))
    df_train, df_valid = train_test_split(df, test_size=0.2)

    # Build the model using modern Keras API


    model = models.Sequential()
    model.add(layers.Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))  # Normalization
    model.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1164, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1))


    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error')

    # Train model (NOTE: fit_generator is deprecated → use fit with generator)
    history = model.fit(
        generate_samples(df_train, local_data_path),
        steps_per_epoch=len(df_train),
        epochs=30,
        validation_data=generate_samples(df_valid, local_data_path, augment=False),
        validation_steps=len(df_valid),
        callbacks=[WeightsLogger(root_path=local_project_path)]
    )

    # Save model JSON
    with open(os.path.join(local_project_path, 'model.json'), 'w') as f:
        f.write(model.to_json())

    # Save weights
    model.save_weights(os.path.join(local_project_path, 'model.h5'))

    backend.clear_session()
