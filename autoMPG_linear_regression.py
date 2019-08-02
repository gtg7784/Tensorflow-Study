from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class printDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print("")
        print(".", end="")


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    plt.figure(figsize=(8, 12))

    plt.subplot(2, 1, 1)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Abs Error [MPG]")
    plt.plot(
        hist["epoch"], hist["mean_absolute_error"],
        label="Train Error"
    )
    plt.ylim([0, 5])
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Square Error [$MPG^2$]")
    plt.plot(
        hist["epoch"], hist["mean_squared_error"],
        label="Train error"
    )
    plt.plot(
        hist["epoch"], hist["val_mean_squared_error"],
        label="Val error"
    )
    plt.ylim([0, 20])
    plt.legend()

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

row_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)

dataset = row_dataset.copy()

dataset = dataset.dropna()
origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

def norm(dataset):
    return (dataset - train_stats['mean']) / train_stats['std']
    
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[9]),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
])

model.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.RMSprop(0.001),
    metrics=['mean_absolute_error', 'mean_squared_error']
)

# patience 매개변수는 성능 향상을 체크할 에포크 횟수입니다
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[early_stop, printDot()]
)

plot_history(history)

Loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.figure()
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

plt.show()