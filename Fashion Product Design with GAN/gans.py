# =========================================================
# KURULUM VE HAZIRLIK ADIMLARI
# =========================================================
# python -m venv venv
# .\venv\Scripts\activate
# pip install tensorflow matplotlib
# pip freeze > requirements.txt

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import fashion_mnist

# =========================================================
# HİPERPARAMETRELER VE VERİ YÜKLEME
# =========================================================
BUFFER_SIZE = 60000        # Veri karıştırma tampon boyutu
BATCH_SIZE = 128           # Her eğitim adımında işlenecek görüntü sayısı
NOISE_DIM = 100            # Generator'a girecek rastgele gürültü vektörünün boyutu
IMAGE_SHAPE = (28, 28, 1)  # Fashion MNIST görüntü boyutu (Gri tonlamalı)
EPOCHS = 20                # Eğitim tur sayısı

# Veri setini yükle (Etiketlere ihtiyacımız yok, sadece görüntüler)
(train_images, _), (_, _) = fashion_mnist.load_data()

# Görüntüleri modelin anlayacağı formata getir
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')

# Normalizasyon: Görüntü piksellerini [0, 255] aralığından [-1, 1] aralığına çekiyoruz.
# Bunun nedeni Generator'ın çıkış katmanında 'tanh' aktivasyon fonksiyonu kullanılmasıdır.
train_images = (train_images - 127.5) / 127.5 

# Veriyi batch'lere ayır ve karıştır
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# =========================================================
# GENERATOR (ÜRETİCİ) MODELİ
# =========================================================
# Amacı: Rastgele gürültüden (noise) gerçekçi görüntüler oluşturmak.
# Yapısı: Küçük bir vektörü alıp, Conv2DTranspose katmanları ile büyüterek (upsampling) resme dönüştürür.
def make_generator_model():
    model = tf.keras.Sequential([
        # 1. Katman: Gürültü vektörünü alıp çok sayıda nörona genişletir
        layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Vektörü 3 boyutlu görüntü formatına çevir (7x7 boyutunda 256 kanal)
        layers.Reshape((7, 7, 256)),

        # 2. Katman: Görüntü boyutunu 7x7'den 14x14'e çıkarır (Upsampling)
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # 3. Katman: Görüntü boyutunu 14x14'e çıkarır (Strides=2 büyütmeyi sağlar)
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # 4. Çıkış Katmanı: Görüntüyü 28x28 boyutuna getirir.
        # Activation 'tanh' sonucun -1 ile 1 arasında olmasını sağlar.
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])

    return model

# =========================================================
# DISCRIMINATOR (AYIRT EDİCİ) MODELİ
# =========================================================
# Amacı: Kendisine gelen görüntünün "Gerçek" mi yoksa "Sahte" (Generator üretimi) mi olduğunu anlamak.
# Yapısı: Standart bir CNN sınıflandırıcıdır.
def make_discriminator_model():
    model = tf.keras.Sequential([
        # 1. Konvolüsyon Bloğu (Downsampling)
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=IMAGE_SHAPE),
        layers.LeakyReLU(),
        layers.Dropout(0.3), # Overfitting'i önlemek için bazı nöronları kapat

        # 2. Konvolüsyon Bloğu
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        # 3. Konvolüsyon Bloğu
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        # Çıkış Katmanı: Görüntüyü düzleştir ve tek bir skor üret
        # Pozitif değerler "Gerçek", Negatif değerler "Sahte" anlamına gelir.
        layers.Flatten(),
        layers.Dense(1)
    ])

    return model


# =========================================================
# KAYIP FONKSİYONLARI VE OPTİMİZASYON
# =========================================================

# from_logits=True: Modelin son katmanında aktivasyon fonksiyonu (sigmoid) olmadığı için
# ham skorları (logits) olasılığa çevirme işini loss fonksiyonuna bırakıyoruz. Bu daha kararlıdır.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    """
    Discriminator'ın amacı:
    1. Gerçek resimleri (real_output) 1'e yaklaştırmak.
    2. Sahte resimleri (fake_output) 0'a yaklaştırmak.
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    """
    Generator'ın amacı:
    Discriminator'ı kandırmak. Yani ürettiği sahte resimler için
    Discriminator'ın "Bu gerçek (1)" demesini sağlamak.
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Modelleri oluştur
generator = make_generator_model()
discriminator = make_discriminator_model()

# Optimizasyon algoritmaları (Adam)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Eğitim sürecini görselleştirmek için sabit bir gürültü (seed) oluşturuyoruz.
# Böylece her epoch sonunda aynı gürültüden nasıl resimler oluştuğunu görebiliriz.
seed = tf.random.normal([16, NOISE_DIM])

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def generate_and_save_images(model, epoch, test_input):
    # Training=False çünkü BatchNormalization eğitim ve testte farklı davranır
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # Görüntüyü [-1, 1] aralığından [0, 255] aralığına geri çekip çizdiriyoruz
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    # Klasör yoksa oluştur
    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')
    
    plt.savefig('generated_images/image_at_epoch_{:03d}.png'.format(epoch))
    plt.close()

# =========================================================
# EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
# =========================================================
def train(dataset, epochs):
    print("Eğitim başlıyor...")
    for epoch in range(1, epochs + 1):
        gen_loss_total = 0
        disc_loss_total = 0
        batch_count = 0

        for image_batch in dataset:
            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

            # GradientTape işlemleri kaydeder ve türev (gradient) hesaplar
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # 1. Generator sahte resim üretir
                generated_images = generator(noise, training=True)

                # 2. Discriminator hem gerçek hem sahte resimleri değerlendirir
                real_output = discriminator(image_batch, training=True)
                fake_output = discriminator(generated_images, training=True)

                # 3. Kayıplar (Loss) hesaplanır
                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            # 4. Gradyanlar hesaplanır (Hata geriye yayılımı - Backpropagation)
            gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            # 5. Ağırlıklar güncellenir
            generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

            gen_loss_total += gen_loss
            disc_loss_total += disc_loss
            batch_count += 1

        # Her epoch sonu durum raporu
        print(f'Epoch {epoch}, Gen Loss: {gen_loss_total/batch_count:.3f}, Disc Loss: {disc_loss_total/batch_count:.3f}')
        
        # Gelişimi görmek için resim kaydet
        generate_and_save_images(generator, epoch, seed)

# Eğitimi Başlat
train(train_dataset, EPOCHS)