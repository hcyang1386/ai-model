import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def load_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)[0]

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

    cam = np.maximum(cam, 0)
    heatmap = cam / tf.reduce_max(cam)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))

    return heatmap

def drawHeatmap(filename):
  model = VGG16(weights='imagenet')
  img_path = f'{filename}'
  img_array = load_image(img_path)
  heatmap = grad_cam(model, img_array, 'block5_conv3')
  return heatmap
  
def drawGradCAM(filename):
  heatmap = drawHeatmap(filename)
  import matplotlib.pyplot as plt
  img = plt.imread(f'{filename}')
  plt.matshow(heatmap)
  plt.imshow(img, alpha=0.4)
  plt.grid()
  plt.show()
  return img, heatmap

def representation(filename, w, h):
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(figsize=(10,10))
  img = plt.imread(f'{filename}')
  plt.imshow(img, alpha=0.5)
  ax.set_xticks(range(0, w))
  ax.set_yticks(range(0, h))

  plt.grid()
  plt.show()

  return img

def sizeImage(fig):
  size_in_inches = fig.get_size_inches()
  dpi = fig.dpi
  size_in_pixels = size_in_inches * dpi
  return size_in_pixels

