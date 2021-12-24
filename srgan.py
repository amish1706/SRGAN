import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (Conv2D, BatchNormalization, LeakyReLU, Add, Flatten, Dense, PReLU, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import numpy as np
from tensorflow.keras.applications import VGG19
import os
import matplotlib.pyplot as plt
# from tensorflow.keras.applications import VGG19
import time
from data_loader import DataLoader
from config import config


class residual_block:
    def __init__(self):
        self.conv = Conv2D(64, (3,3), padding='same')
        self.batch_norm = BatchNormalization()
        self.activation = PReLU()
        self.add = Add()
        
    def __call__(self, inputs):
        temp = inputs
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.add([x, temp])
        return x

def Generator(inp, b=16):
    x = Conv2D(64,(9,9), padding='same')(inp)
    x = PReLU()(x)
    temp = x

    for i in range(b):
        x = residual_block()(x)
    
    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, temp])

    x = Conv2D(256,(3,3), padding='same')(x)
    x = UpSampling2D(size=(2,2))(x)
    x = PReLU()(x)
    x = Conv2D(256,(3,3), padding='same')(x)
    x = UpSampling2D(size=(2,2))(x)
    x = PReLU()(x)

    x = Conv2D(3,(9,9), padding='same')(x)

    return x

def Discriminator(inp):
    x = Conv2D(64,(3,3), padding='same')(inp)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(64,(3,3),(2,2),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128,(3,3),(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128,(3,3),(2,2),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256,(3,3),(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256,(3,3),(2,2),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512,(3,3),(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512,(3,3),(2,2),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return x

class SRGAN():
    def __init__(self,lr,batch_size, epochs, lr_decay):
        # decay_every = epochs//2
        self.epochs=epochs
        self.batch_size=batch_size
        self.channels = 3
        self.lr_height = 56                 # Low resolution height
        self.lr_width = 56                  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height*4   # High resolution height
        self.hr_width = self.lr_width*4     # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # self.generator = self.build_generator()
        # self.discriminator = self.build_discriminator()
        self.vgg = self.build_vgg()

        self.gen_optimizer = tf.keras.optimizers.Adam(lr)
        self.disc_optimizer = tf.keras.optimizers.Adam(lr)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()

        self.checkpoint_dir = 'training_checkpoints'
        

    def build_vgg(self):
        vgg = VGG19(weights='imagenet')
        # img = Input(shape=self.hr_shape)
        output = [vgg.layers[9].output]
        vgg.trainable=False
        # img_features = vgg(img)
        return Model(vgg.input, output)

    def build_generator(self):
        img_lr = Input(shape=self.lr_shape)
        gen_hr = Generator(img_lr)
        return Model(img_lr, gen_hr)
    
    def build_discriminator(self):
        inp = Input(shape=self.hr_shape)
        validity = Discriminator(inp)
        return Model(inp, validity)

    def discriminator_loss(self,og_hr_output, gen_hr_output):
        real_loss = self.cross_entropy(tf.ones_like(og_hr_output), og_hr_output)
        fake_loss = self.cross_entropy(tf.zeros_like(gen_hr_output), gen_hr_output)
        total_loss = real_loss + fake_loss
        return total_loss 

    def generator_loss(self,gen_hr_output):
        return self.cross_entropy(tf.ones_like(gen_hr_output), gen_hr_output)

    def perceptual_loss(self, gen_features, og_features,gen_hr_output):
        gen_loss = self.generator_loss(gen_hr_output)
        content_loss = tf.keras.losses.MeanAbsoluteError()(gen_features,og_features)
        return content_loss + gen_loss*(1e-3)
    
    @tf.function
    def train_step(self,img_lr, img_hr, generator, discriminator):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_hr = generator(img_lr)
            og_hr_output = discriminator(img_hr)
            gen_hr_output = discriminator(gen_hr)
            gen_features = self.vgg((gen_hr+1)/2)
            og_features = self.vgg((img_hr+1)/2)
            disc_loss = self.discriminator_loss(og_hr_output, gen_hr_output)
            perc_loss = self.perceptual_loss(gen_features, og_features, gen_hr_output)
            
        grads_gen = gen_tape.gradient(perc_loss,generator.trainable_variables)
        grads_disc = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(grads_gen, generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_variables))
        
    
    def train(self, from_last_checkpoint=True):
        Generator = self.build_generator()
        Discriminator = self.build_discriminator()
        if from_last_checkpoint:
            prev_epoch = input("Enter prev_epoch number:")
            Generator.load_weights("./training_checkpoints/epoch_{}_g.h5".format(prev_epoch))
            Discriminator.load_weights("./training_checkpoints/epoch_{}_d.h5".format(prev_epoch))
        loader = DataLoader('train', self.hr_shape)
        train_ds = loader.load_data(batch_size=self.batch_size)
        for epoch in range(1, self.epochs+1):
            start = time.time()
            print("Epoch: {}".format(epoch))
            bar = tf.keras.utils.Progbar(len(list(train_ds.as_numpy_iterator())))
            for idx,train_imgs in enumerate(train_ds):
                img_lr = train_imgs[0]
                img_hr = train_imgs[1]
                self.train_step(img_lr,img_hr, Generator, Discriminator)
                bar.update(idx+1)
            if (epoch) % 5 == 0 or epoch == self.epochs:
                Generator.save_weights(os.path.join(self.checkpoint_dir, 'epoch_{}_g.h5'.format(int(prev_epoch)+epoch)))
                Discriminator.save_weights(os.path.join(self.checkpoint_dir, 'epoch_{}_d.h5'.format(int(prev_epoch)+epoch)))

            print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
    
    def generate(self, val_batch_size=2):
        loader = DataLoader('val',self.hr_shape)
        val_ds = loader.load_data(batch_size=val_batch_size)
        Gen = self.build_generator()
        Gen.load_weights(os.path.join(self.checkpoint_dir, 'epoch_{}_g.h5'.format(self.epochs)))
        x = 0
        for val_imgs in val_ds:
            gen_imgs = Gen(val_imgs[0])
            for i in range(val_batch_size):
                img_lr = val_imgs[0][i]
                img_hr = val_imgs[1][i]
                gen_img = gen_imgs[i]
                fig = plt.figure(figsize=(1,3))
                fig.set_figwidth(20)
                fig.set_figheight(20)
                plt.subplot(1,3,1)
                plt.imshow(img_lr)
                plt.subplot(1,3,2)
                plt.imshow(img_hr)
                plt.subplot(1,3,3)
                plt.imshow(gen_img)
                plt.savefig("./generated_images/gen_{}.jpg".format(x))
                x+=1

        # val_data = val_ds.as_numpy_iterator()
        # for n in range(num_eg):
        #     fig = plt.figure(figsize=(1,3))
        #     fig.set_figwidth(20)
        #     fig.set_figheight(20)
        #     imgs = val_data.next()
        #     img_lr = imgs[0][0]
        #     img_hr = imgs[1][0]
        #     gen_hr = Gen(img_lr)
        #     plt.subplot(1,3,1)
        #     plt.imshow(img_lr)
        #     plt.subplot(1,3,2)
        #     plt.imshow(img_hr)
        #     plt.subplot(1,3,3)
        #     plt.imshow(gen_hr)
        #     plt.savefig("gen_{}.jpg".format(n))

        # for val_imgs in val_ds:
        #     for images in val_imgs:
        #         img_lr,img_hr = images
        #         gen_hr = Gen(img_lr)
        #         plt.subplot(1,2,1)
        #         plt.imshow(gen_hr[0])
        #         plt.subplot(1,2,2)
        #         plt.imshow(img_hr[0])
        #         plt.axis('off')
        #         plt.savefig("gen_"+str(i+1))
        #         i+=1

LR = config.TRAIN.lr           
BATCH_SIZE = config.TRAIN.batch_size
EPOCHS =  config.TRAIN.n_epoch
LR_DECAY = config.TRAIN.lr_decay

if __name__ == "__main__":
    gan = SRGAN(LR, BATCH_SIZE, EPOCHS, LR_DECAY)
    gan.train()