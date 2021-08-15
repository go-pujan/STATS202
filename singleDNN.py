import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

sns.set_theme(style="ticks")
df = pd.read_csv("A_train_data.csv")
a = df[df['time'] == "10:28:05"]

train_dataset = a.sample(frac=0.8, random_state=0)
test_dataset = a.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('open')
test_labels = test_features.pop('open')

print(train_dataset.describe().transpose()[['mean', 'std']])
day = np.array(train_features['day'])

day_normalizer = preprocessing.Normalization(input_shape=[1, ], axis=None)
day_normalizer.adapt(day)


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

    dnn_day_model = build_and_compile_model(day_normalizer)
    dnn_day_model.summary()
