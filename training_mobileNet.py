import tensorflow as tf
from tensorflow.keras import layers
import time
import matplotlib.pyplot as plt
import numpy as np

batch_size = 16
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
  "/content/spectrograms_final/train",
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds  = tf.keras.utils.image_dataset_from_directory(
  "/content/spectrograms_final/val",
  image_size=(img_height, img_width),
  batch_size=batch_size)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Define the MobileNet model
def mobilenet():
    input_shape = (224, 224, 3)

    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise separable convolutions
    x = layers.DepthwiseConv2D((3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(5):
        x = layers.DepthwiseConv2D((3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(512, (1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(1024, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)

    # Output layer
    outputs = layers.Dense(13, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


model = mobilenet()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs["val_accuracy"]
        if val_accuracy >= self.threshold:
            self.model.stop_training = True

my_callback = MyThresholdCallback(threshold=0.9)

start = time.time()
history=model.fit(
  x=train_ds,
  validation_data=val_ds,
  epochs=25
)
end = time.time()
print('Time: ', end-start)

model.save('mobileNet_25epochs.keras')

prediction = np.array(model.predict(val_ds))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
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




