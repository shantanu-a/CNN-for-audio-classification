import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers
import time


batch_size = 16
img_height = 224
img_width = 224

TRAIN_DS_DIRECTORY = "content/spectrograms/train/"
TEST_DS_DIRECTORY = "content/spectrograms/val/"


train_ds = tf.keras.utils.image_dataset_from_directory(
  TRAIN_DS_DIRECTORY,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  TEST_DS_DIRECTORY,
  image_size=(img_height, img_width),
  batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

def convolutional_block(x, filter):
    x_skip = x

    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    x_skip = tf.keras.layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)

    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def identity_block(x, filter):
    x_skip = x

    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def resnet(input_shape=(224, 224, 3), classes=13):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = layers.ZeroPadding2D((3, 3))(x)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    block_layers = [2, 2, 2, 2]
    filter_size = 64

    for i in range(4):
        if i == 0:
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)

    x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    # x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dense(classes)(x)

    model = tf.keras.Model(inputs, x)
    return model

model = resnet()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs["val_accuracy"]
        if val_accuracy >= self.threshold:
            self.model.stop_training = True

my_callback = MyThresholdCallback(threshold=0.87)


start = time.time()
hist = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=30,
  callbacks = [my_callback]
)
end = time.time()
print('Time: ', end-start)

model.save('resnetfinal.keras')

prediction = np.array(model.predict(val_ds))

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()

from sklearn.metrics import precision_recall_fscore_support
val_images, val_labels = zip(*val_ds.unbatch().as_numpy_iterator())
val_labels = np.array(val_labels)

predicted_labels = prediction.argmax(axis=1)
precision_recall_fscore_support(val_labels, predicted_labels)

tp = 0
fn = 0

for x, y in val_ds:
  y = np.array(y)
  predic = np.argmax(predic, axis=1)
  # print(np.argmax(predic, axis=1))
  for i in range(len(y)):
    if y[i] == predic[i]:
      tp += 1
    else:
      fn += 1

print(f'Overall Precision: {tp / (tp + fn)}')

