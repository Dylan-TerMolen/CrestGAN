# Commented out IPython magic to ensure Python compatibility.
try:
  %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

path = "drive/My Drive/Logos_Directory/ClubLogos"
my_dir = [os.path.join(path,logo_file) for logo_file in os.listdir(path)]



cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

from google.colab import drive
drive.mount('/content/drive')

def create_generator():
  model = tf.keras.Sequential()
  model.add(layers.Dense(8*8*1028, use_bias=False, input_shape=(100,)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((8, 8, 1028)))

  model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 16, 16, 512)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 32, 32, 256)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 64, 64, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 128, 128, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  
  model.add(layers.Conv2DTranspose(3,(5,5),strides=(1,1),padding='same',use_bias=False))
  assert model.output_shape == (None, 128, 128, 3)

  return model


def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)

def create_discriminator():
  model = tf.keras.Sequential()
  
  model.add(layers.Conv2D(64,(5,5),strides=(2,2),padding='same',input_shape=(128,128,3)))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(128,(5,5),strides=(2,2),padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.BatchNormalization())

  model.add(layers.Conv2D(256,(5,5),strides=(2,2),padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.BatchNormalization())
  
  model.add(layers.Conv2D(512,(5,5),strides=(2,2),padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.BatchNormalization())

  model.add(layers.Conv2D(1024,(5,5),strides=(2,2),padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.BatchNormalization())

  model.add(layers.Flatten())
  model.add(layers.Dense(1))
  
  return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
    
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

my_generator = create_generator()
my_discriminator = create_discriminator()


checkpoint_dir = './drive/My Drive/GAN_Training_Checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=my_generator,
                                 discriminator=my_discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 4

seed = tf.random.normal([num_examples_to_generate, noise_dim])

def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = my_generator(noise, training=True)

      real_output = my_discriminator(images, training=True)
      fake_output = my_discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, my_generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, my_discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, my_generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, my_discriminator.trainable_variables))

    return gen_loss,disc_loss

def preprocess_image(file_path,shapeX=128,shapeY=128):
    img_raw = tf.io.read_file(file_path)
    img_tensor = tf.image.decode_jpeg(img_raw,channels=3)
    img_final = tf.image.resize(img_tensor, [shapeX, shapeY])
    img_normalized = img_final / 255.0
    return img_normalized


#%%
my_machine_path = 'drive/My Drive/Logos_Directory/ClubLogos'

logos_dir_with_path = [os.path.join(my_machine_path,logo_file) for logo_file in os.listdir(my_machine_path)]
logos_dir_with_path.remove(os.path.join(my_machine_path,'.DS_Store'))

BUFFER_SIZE = 60000
BATCH_SIZE = 64

path_ds = tf.data.Dataset.from_tensor_slices(logos_dir_with_path)
logo_ds = path_ds.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds = logo_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

GAseed = tf.random.normal([1, 100])
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    g_loss = []
    d_loss = []
    for image_batch in dataset:
      g_loss_temp,d_loss_temp = train_step(image_batch)
      g_loss.append(g_loss_temp)
      d_loss.append(d_loss_temp)
    genLossAvg = np.average(g_loss)
    discLossAvg = np.average(d_loss)
    epoch_summary = "Epoch {}: Generator Loss = {}, Discriminator Loss = {}".format(str(epoch + 1),str(genLossAvg),str(discLossAvg))
    print(epoch_summary)
    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


    # Save the model every 15 epochs
    if (epoch + 1) % 30 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    if epoch % 100 == 0:
      print("Seed is first picture, random is second")
      img = my_generator(GAseed,training=False)
      plt.imshow(img[0,:,:,:])
      plt.show()
      print_ten_images()
def print_ten_images():
  for i in range(10):
    img = my_generator(tf.random.normal([1,100]),training=False)
    print(my_discriminator(img,training=False))
    plt.imshow(img[0,:,:,:])
    plt.show()


train(ds,1000)


