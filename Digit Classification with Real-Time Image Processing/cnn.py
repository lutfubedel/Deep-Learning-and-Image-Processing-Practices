# ==========================================
# KURULUM VE HAZIRLIK KOMUTLARI (Terminal)
# ==========================================

# python -m venv venv
# .\venv\Scripts\activate
# pip install opencv-python tensorflow matplotlib
# pip freeze > requirements.txt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ==========================================
# 1. VERİ YÜKLEME VE GÖRSELLEŞTİRME
# ==========================================

# MNIST veri setini (el yazısı rakamlar) yükle
# Eğitim (train) ve Test setlerine ayır
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# DİKKAT: MNIST normalde siyah arka plan üzerine beyaz yazıdır.
# Aşağıdaki işlem renkleri tersine çevirir (Beyaz arka plan üzerine siyah yazı).
# Bu, genelde kağıda çizip kameradan okuttuğumuz görsellerle modelin uyumlu olması için yapılır.
x_train = 255 - x_train
x_test = 255 - x_test

# Veriyi kontrol etmek için ilk 5 örneği çizdir
plt.figure(figsize=(10,5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i], cmap='gray') # Gri tonlamalı göster
    plt.axis('off') # Eksenleri kapat
    plt.title(f'Etiket: {y_train[i]}')

plt.tight_layout()
plt.show()

# ==========================================
# 2. VERİ ÖN İŞLEME (PREPROCESSING)
# ==========================================

# Reshape: Veriyi CNN'in kabul edeceği formata sokma.
# (Adet, Genişlik, Yükseklik, Kanal Sayısı) -> (-1, 28, 28, 1)
# 1 kanalı (channel) gri tonlama (grayscale) olduğu anlamına gelir.
# Normalizasyon: Piksel değerlerini 0-255 arasından 0-1 arasına sıkıştırma (işlem hızını ve başarımı artırır).
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# Data Augmentation (Veri Çoğaltma/Zenginleştirme):
# Modelin ezberlemesini (overfitting) önlemek için mevcut resimleri hafifçe değiştirerek
# eğitim setini yapay olarak çeşitlendiriyoruz.
datagen = ImageDataGenerator(
    rotation_range=10,      # Rastgele 10 derece döndür
    zoom_range=0.1,         # Rastgele %10 yakınlaştır/uzaklaştır
    width_shift_range=0.1,  # Yatayda %10 kaydır
    height_shift_range=0.1  # Dikeyde %10 kaydır
)

# ==========================================
# 3. MODEL MİMARİSİ (CNN)
# ==========================================

model = models.Sequential([
    # -- Özellik Çıkarma (Feature Extraction) Kısmı --
    
    # 1. Konvolüsyon Katmanı: Resimdeki kenar, köşe gibi özellikleri yakalar.
    # 32 filtre kullanır, 3x3 boyutunda tarayıcı filtreler gezdirir.
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # 1. Havuzlama (Pooling): Resim boyutunu yarıya indirir (28x28 -> 14x14).
    # En belirgin özellikleri tutar, işlem yükünü azaltır.
    layers.MaxPooling2D((2, 2)),

    # 2. Konvolüsyon Katmanı: Daha karmaşık özellikleri (yuvarlaklar, çizgiler) yakalar.
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # 2. Havuzlama: Boyutu tekrar yarıya indirir (Küçülmeye devam eder).
    layers.MaxPooling2D((2, 2)),

    # -- Sınıflandırma (Classification) Kısmı --

    # Flatten: 2 boyutlu matrisleri (kare resimleri) tek boyutlu bir vektöre (diziye) çevirir.
    # Dense katmanına girmeden önce düzleştirme şarttır.
    layers.Flatten(),
    
    # Tam Bağlantılı (Dense) Katman: Öğrenilen özellikleri yorumlar.
    layers.Dense(64, activation='relu'),
    
    # Çıkış Katmanı: 10 adet nöron vardır (0'dan 9'a kadar rakamlar için).
    # Softmax: Çıktıları olasılığa dönüştürür (Toplamı 1 olur). En yüksek olasılık tahmindir.
    layers.Dense(10, activation='softmax')
])

# Model özetini tablo olarak göster
print(model.summary())

# ==========================================
# 4. DERLEME VE EĞİTİM (COMPILE & TRAIN)
# ==========================================

# Optimizer: 'adam' (Genelde en iyi performansı veren adaptif bir optimizasyon algoritması)
# Loss: 'sparse_categorical_crossentropy' (Etiketlerimiz 0,1,2 gibi tamsayı olduğu için bunu kullanıyoruz)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) # Başarıyı 'doğruluk' üzerinden takip et

# Modeli Eğitme
# datagen.flow: Zenginleştirilmiş veriyi parça parça (batch) modele besler.
# validation_data: Her epoch (tur) sonunda modelin hiç görmediği test verisiyle başarısını ölçer.
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=10, 
                    validation_data=(x_test, y_test))

# Modeli kaydet (Daha sonra tekrar eğitmek zorunda kalmadan kullanmak için)
model.save('mnist_cnn_model.h5')
print("Model kaydedildi: mnist_cnn_model.h5")