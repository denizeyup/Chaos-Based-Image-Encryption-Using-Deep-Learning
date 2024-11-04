import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Activation, BatchNormalization, LeakyReLU, Add, Flatten, Dense, Dropout

import tensorflow.keras.backend as K

# Constants for SSIM
L = 255.0  # maximum pixel value
k1 = 0.01
k2 = 0.03
C1 = (k1 * L) ** 2
C2 = (k2 * L) ** 2
mu = 0.2  # hyperparameter for balancing SSIM in loss calculations

def ssim(x, y):
    ux = tf.reduce_mean(x)
    uy = tf.reduce_mean(y)
    stdx = tf.math.reduce_std(x)
    stdy = tf.math.reduce_std(y)
    stdxy = tf.reduce_mean((x - ux) * (y - uy))
    ssim = ((2 * ux * uy + C1) * (2 * stdxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (stdx ** 2 + stdy ** 2 + C2))
    return ssim

def discriminator_loss(y_true, y_pred):
    return mu * (1 - ssim(y_true, y_pred))

# SSIM lost function
def ssim_loss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val=255)
    return 1 - K.mean(ssim)

# InstanceNormalization
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale',
                                     shape=input_shape[-1:],
                                     initializer=tf.random_normal_initializer(1., 0.02),
                                     trainable=True)
        self.offset = self.add_weight(name='offset',
                                      shape=input_shape[-1:],
                                      initializer='zeros',
                                      trainable=True)
        super(InstanceNormalization, self).build(input_shape)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

# Adding Noise
def add_noise(image, noise_factor=0.3):
    row, col, ch = image.shape
    mean = 0
    sigma = noise_factor * 255
    
    gauss = np.random.normal(mean, sigma, (row, col, 1))
    
    gauss = np.concatenate([gauss] * ch, axis=-1)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

def load_and_resize_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        image = cv2.resize(image, target_size)
        image = np.expand_dims(image, axis=-1)
        return image
    else:
        print(f"Hata: {image_path} yolundaki görüntü yüklenemedi.")
        return None

# Build Generator
def build_generator():

    def residual_block(x, filters):
        res = x
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.3)(x)
        x = Add()([x, res])
        x = Activation('relu')(x)
        return x

    input_layer = Input(shape=(256, 256, 1))
    conv1 = Conv2D(64, (7, 7), padding='same')(input_layer)
    conv1 = InstanceNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(128, (3, 3), padding='same')(conv1)
    conv2 = InstanceNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(256, (3, 3), padding='same')(conv2)
    conv3 = InstanceNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = residual_block(conv3, 256)
    conv3 = residual_block(conv3, 256)
    tconv1 = Conv2DTranspose(256, (3, 3), padding='same')(conv3)
    tconv1 = InstanceNormalization()(tconv1)
    tconv1 = Activation('relu')(tconv1)
    tconv2 = Conv2DTranspose(128, (3, 3), padding='same')(tconv1)
    tconv2 = InstanceNormalization()(tconv2)
    tconv2 = Activation('relu')(tconv2)
    tconv3 = Conv2DTranspose(64, (3, 3), padding='same')(tconv2)
    tconv3 = InstanceNormalization()(tconv3)
    tconv3 = Activation('relu')(tconv3)
    tconv3 = residual_block(tconv3, 64)
    tconv3 = residual_block(tconv3, 64)
    conv4 = Conv2D(64, (7, 7), padding='same')(tconv3)
    conv4 = InstanceNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv5 = Conv2D(128, (3, 3), padding='same')(conv4)
    conv5 = InstanceNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = residual_block(conv5, 128)
    conv5 = residual_block(conv5, 128)
    tconv4 = Conv2DTranspose(128, (3, 3), padding='same')(conv5)
    tconv4 = InstanceNormalization()(tconv4)
    tconv4 = Activation('relu')(tconv4)
    tconv5 = Conv2DTranspose(64, (3, 3), padding='same')(tconv4)
    tconv5 = InstanceNormalization()(tconv5)
    tconv5 = Activation('relu')(tconv5)
    tconv5 = residual_block(tconv5, 64)
    tconv5 = residual_block(tconv5, 64)
    conv6 = Conv2D(64, (7, 7), padding='same')(tconv5)
    conv6 = InstanceNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv7 = Conv2D(128, (3, 3), padding='same')(conv6)
    conv7 = InstanceNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv8 = Conv2D(256, (3, 3), padding='same')(conv7)
    conv8 = InstanceNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = residual_block(conv8, 256)
    conv8 = residual_block(conv8, 256)
    tconv6 = Conv2DTranspose(256, (3, 3), padding='same')(conv8)
    tconv6 = InstanceNormalization()(tconv6)
    tconv6 = Activation('relu')(tconv6)
    tconv7 = Conv2DTranspose(128, (3, 3), padding='same')(tconv6)
    tconv7 = InstanceNormalization()(tconv7)
    tconv7 = Activation('relu')(tconv7)
    tconv8 = Conv2DTranspose(64, (3, 3), padding='same')(tconv7)
    tconv8 = InstanceNormalization()(tconv8)
    tconv8 = Activation('relu')(tconv8)
    conv7x7 = Conv2D(1, (7, 7), padding='same')(tconv8)
    output_layer = Activation('relu')(conv7x7)

    generator = Model(inputs=input_layer, outputs=output_layer, name='Generator')
    generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999),
                      loss="mse")
    return generator

# Build Discriminator
def build_discriminator():
    input_layer = Input(shape=(256, 256, 1))
    x = Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation=LeakyReLU(0.2))(input_layer)
    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(1, (4, 4), padding='same')(x)
    x = Flatten()(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer, name='Discriminator')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999), loss=discriminator_loss)
    return model

# Build GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    input_layer = Input(shape=(256, 256, 1))
    generated_image = generator(input_layer)
    gan_output = discriminator(generated_image)
    gan = Model(inputs=input_layer, outputs=gan_output, name='GAN')
    gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999),
                loss=discriminator_loss)
    return gan

generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Data Loading
train_path = "path/your/dataset"
image_files = [os.path.join(train_path, file) for file in os.listdir(train_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
x_train = [cv2.imread(image_file, cv2.IMREAD_GRAYSCALE) for image_file in image_files if cv2.imread(image_file, cv2.IMREAD_GRAYSCALE) is not None]
x_train = np.array(x_train)

# Model Train Parameter
epochs = 10
batch_size = 8

plain_image_path = "dataset/image.jpg"
plain_image = load_and_resize_image(plain_image_path)
plain_image = np.expand_dims(plain_image, axis=0) 
print("Giriş görüntüsünün boyutu:", plain_image.shape)

for epoch in range(epochs):
    for _ in range(batch_size):
       
        plain_image_ = add_noise(plain_image)
        plain_image_ = tf.image.resize(plain_image_, size=(256, 256))

        
        generated_image = generator.predict(plain_image_)
        generated_image = tf.image.resize(generated_image, size=(256, 256))
        generated_image = np.squeeze(generated_image, axis=0)  
        generated_image = np.expand_dims(generated_image, axis=0)  

        random_index = np.random.randint(low=0, high=x_train.shape[0] - batch_size)
        cipher_images = x_train[random_index: random_index + batch_size]
        cipher_images = tf.image.resize(cipher_images, size=(256, 256))

        
        if cipher_images.ndim < 4:
            cipher_images = np.expand_dims(cipher_images, axis=-1)  

        
        x = np.concatenate([cipher_images, generated_image], axis=0)

        
        y_dis = np.zeros((cipher_images.shape[0] + generated_image.shape[0],))
        y_dis[:cipher_images.shape[0]] = 1

        discriminator.trainable = True
        discriminator.train_on_batch(x, y_dis)

        y_gen = np.ones((1,))
        discriminator.trainable = False
        gan.train_on_batch(plain_image_, y_gen)

    print(f"Epoch {epoch + 1}/{epochs} completed")


# Save model
gan.save_weights('gans_model.h5')

# Test
plain_image = cv2.imread(plain_image_path, cv2.IMREAD_GRAYSCALE)
plain_image = np.expand_dims(plain_image, axis=-1)
plain_image = np.expand_dims(plain_image, axis=0)
generated_images = generator.predict(plain_image)
plt.imshow(generated_images[0], cmap='gray')
plt.axis('off')
plt.show()
