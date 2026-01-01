# =========================================================
# Gerekli kütüphanelerin yüklenmesi
# =========================================================
# python -m venv venv
# .\venv\Scripts\activate
# pip install tensorflow==2.13.0 tensorflow-datasets==4.9.2 matplotlib opencv-python
# pip freeze > requirements.txt

from tensorflow_datasets import load              # Hazır veri setlerini kolayca indirmek için
from tensorflow.data import AUTOTUNE               # Performans optimizasyonu için otomatik ayar
from tensorflow.keras.models import Sequential     # Katmanları sıralı şekilde tanımlamak için
from tensorflow.keras.layers import (
    Conv2D,       # Evrişim (Convolution) katmanı
    MaxPooling2D, # Havuzlama (Pooling) katmanı
    Flatten,      # Özellik haritalarını vektöre dönüştürür
    Dense,        # Tam bağlantılı (Fully Connected) katman
    Dropout       # Overfitting’i azaltmak için
)
from tensorflow.keras.optimizers import Adam       # Adaptif öğrenme oranına sahip optimizer
from tensorflow.keras.callbacks import (
    EarlyStopping,        # Aşırı öğrenmeyi önler
    ModelCheckpoint,      # En iyi modeli diske kaydeder
    ReduceLROnPlateau     # Öğrenme oranını dinamik olarak düşürür
)

import tensorflow as tf
import matplotlib.pyplot as plt


# =========================================================
# Veri setinin yüklenmesi
# =========================================================
# tf_flowers veri seti 5 sınıftan oluşur:
# Daisy, Dandelion, Roses, Sunflowers, Tulips

(ds_train, ds_val), ds_info = load(
    "tf_flowers", 
    split=["train[:80%]", "train[80%:]"],  # %80 eğitim, %20 doğrulama
    as_supervised=True,                    # (image, label) formatında getirir
    with_info=True                         # Veri seti hakkında meta bilgi sağlar
)

# Veri setinin özelliklerini yazdır
print(ds_info.features)
print("Number of classes:", ds_info.features["label"].num_classes)


# =========================================================
# Örnek görüntülerin görselleştirilmesi
# =========================================================
# Eğitim setinden rastgele 6 görüntü gösterilir
fig = plt.figure(figsize=(10, 5))
for i, (image, label) in enumerate(ds_train.take(6)):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.imshow(image.numpy().astype("uint8"))
    ax.set_title(ds_info.features["label"].int2str(label))
    ax.axis("off")

plt.tight_layout()
plt.show()


# =========================================================
# Data Augmentation & Preprocessing
# =========================================================

# Modelin kabul edeceği standart görüntü boyutu
IMG_SIZE = (180, 180)

def preprocess_train(image, label):
    """
    Eğitim verisi için veri artırma (data augmentation) uygulanır.
    Amaç:
    - Modelin farklı senaryolara karşı daha dayanıklı öğrenmesi
    - Overfitting’i azaltmak
    """
    image = tf.image.resize(image, IMG_SIZE)             # Görüntüyü yeniden boyutlandır
    image = tf.image.random_flip_left_right(image)       # Rastgele yatay çevirme
    image = tf.image.random_brightness(image, 0.2)       # Parlaklık değişimi
    image = tf.image.random_contrast(image, 0.9, 1.2)    # Kontrast değişimi
    image = tf.image.random_crop(image, [160, 160, 3])   # Rastgele kırpma
    image = tf.image.resize(image, IMG_SIZE)             # Tekrar sabit boyuta getir
    image = tf.cast(image, tf.float32) / 255.0           # Normalize et (0-1 arası)
    return image, label

def preprocess_val(image, label):
    """
    Doğrulama verisinde augmentation yapılmaz.
    Amaç:
    - Gerçek dünyaya en yakın performansı ölçmek
    """
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# Eğitim veri setinin hazırlanması
ds_train = (
    ds_train
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)  # Ön işleme
    .shuffle(1000)                                       # Veriyi karıştır
    .batch(32)                                           # Mini-batch
    .prefetch(AUTOTUNE)                                  # GPU/CPU verimliliği
)

# Doğrulama veri setinin hazırlanması
ds_val = (
    ds_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)
    .batch(32)
    .prefetch(AUTOTUNE)
)


# =========================================================
# CNN Modelinin Tanımlanması
# =========================================================
model = Sequential([
    # 1. Convolution Bloğu
    Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),

    # 2. Convolution Bloğu
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # 3. Convolution Bloğu
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Özellik haritalarını vektöre dönüştür
    Flatten(),

    # Tam bağlantılı katman
    Dense(128, activation='relu'),
    Dropout(0.5),  # %50 nöron kapatılır (overfitting önleme)

    # Çıkış katmanı (Softmax çok sınıflı sınıflandırma için)
    Dense(ds_info.features["label"].num_classes, activation='softmax')
])


# =========================================================
# Callback Tanımları
# =========================================================
callbacks = [
    # Validation loss iyileşmezse eğitimi erken durdur
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),

    # Validation loss plato yaparsa öğrenme oranını düşür
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        verbose=1,
        min_lr=1e-9
    ),

    # En iyi modeli diske kaydet
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
]


# =========================================================
# Modelin Derlenmesi (Compile)
# =========================================================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # Etiketler integer olduğu için
    metrics=['accuracy']
)

# Model mimarisini yazdır
print(model.summary())


# =========================================================
# Modelin Eğitilmesi
# =========================================================
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)


# =========================================================
# Eğitim Sonuçlarının Görselleştirilmesi
# =========================================================
plt.figure(figsize=(12, 4))

# Accuracy grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

# Loss grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

plt.show()
