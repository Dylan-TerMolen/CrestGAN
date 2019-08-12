import tensorflow as tf
import numpy as np
import pandas as pd
import os
import PIL.Image
import matplotlib.pyplot as plt
import cv2
import time
print('importing tf depreciation error try to fix ^^^ ')
import models

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# transform an image path into a normalized image tensor that can be processed by a model
def preprocess_image(file_path,shapeX=120,shapeY=120):
    img_raw = tf.io.read_file(file_path)
    img_tensor = tf.image.decode_jpeg(img_raw,channels=3)
    img_final = tf.image.resize(img_tensor, [shapeX, shapeY])
    img_normalized = img_final / 255.0
    return img_normalized

# transform an image directory into a tf Dataset of images as tensors
def preprocess_image_directory(image_dir):
    path_ds = tf.data.Dataset.from_tensor_slices(image_dir)
    image_ds = path_ds.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return image_ds

# transform a list of labels into a tf Dataset of strings
def preprocess_labels(labels_list):
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels_list,tf.int64))
    return label_ds

def preprocess_data(image_dir,labels_list):
    image_dataset = preprocess_image_directory(image_dir)
    label_dataset = preprocess_labels(labels_list)
    image_with_label_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    return image_with_label_dataset


def shuffle_batch_prefetch_repeat(dataset,length_of_dataset,batch_size):
    ds = dataset.shuffle(buffer_size=length_of_dataset)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def test_generator_with_random_vector(my_generator):
    noise = tf.random.normal([1, 100])
    generated_image = my_generator(noise, training=False)
    # reshaped_image = generated_image[0,:,:,:]
    return generated_image

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# @tf.function
def train_step(images,BATCH_SIZE,noise_dim,generator,discriminator,generator_optimizer,discriminator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images[0], training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    print('1 train step')

def train(dataset, epochs,BATCH_SIZE,noise_dim,generator,discriminator,generator_optimizer,discriminator_optimizer):
  for epoch in range(epochs):
    for image_batch in dataset:
        print('training')
        train_step(image_batch,BATCH_SIZE,noise_dim,generator,discriminator,generator_optimizer,discriminator_optimizer)
        test_img = test_generator_with_random_vector(generator)
        plt.imshow(test_img[0,:,:,:])
        plt.show()
        print('trained')
    print("Epoch",epoch,"done")


def main():
    logos_path = '/Users/Dylan/Desktop/ClubLogosJPG'
    logos_dir = os.listdir(logos_path)
    logos_dir_with_path = [logos_path + '/' + fileName for fileName in logos_dir]

    labels = ['logo']
    labels_dict = dict((index, name) for index,name in enumerate(labels))
    labels_list = [0 for i in range(len(logos_dir_with_path))]

    dataset = preprocess_data(logos_dir_with_path,labels_list)
    # processed_dataset = shuffle_batch_prefetch_repeat(dataset,len(labels_list),32)
    # BUFFER_SIZE = 60000


    generator = models.create_generator_model()
    discriminator = models.create_discriminator_model()

    generator_optimizer = models.adam_optimizer()
    discriminator_optimizer = models.adam_optimizer()

    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator,
                                     generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer)
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16
    BATCH_SIZE = 64
    batched_dataset = dataset.batch(BATCH_SIZE)
    seed = tf.random.normal([num_examples_to_generate,noise_dim])
    train(batched_dataset,EPOCHS,BATCH_SIZE,noise_dim,generator,discriminator,generator_optimizer,discriminator_optimizer)
    generator.save('my_logo_generator.h5')

if __name__ == '__main__':
    main()