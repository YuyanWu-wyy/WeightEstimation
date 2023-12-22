import scipy.io
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


file_path = '../weight_music_test/features.mat'
data = scipy.io.loadmat(file_path)
fft_features = data['fft_features']
weight = data['labels']
sensor_num = data['sensor_nums']
fft_features = fft_features[:,1:501]
print(fft_features.shape)
sen = 1
idx = np.where(sensor_num == sen)[0]
features = fft_features[idx,:]
weight = weight[idx, :]

X_train, X_test, y_train, y_test = train_test_split(features, weight, test_size=0.2, random_state=42)
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Single output node for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae}")