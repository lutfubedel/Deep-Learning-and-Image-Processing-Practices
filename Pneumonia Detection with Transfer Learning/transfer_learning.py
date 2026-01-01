# =========================================================
# KURULUM VE HAZIRLIK ADIMLARI
# =========================================================
# python -m venv venv
# .\venv\Scripts\activate
# pip install tensorflow matplotlib scikit-learn
# pip freeze > requirements.txt
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# Gerekli kütüphaneleri içe aktar
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Veri artırma ve yükleme
from tensorflow.keras.applications import DenseNet121              # Transfer Learning için önceden eğitilmiş model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # Katmanlar
from tensorflow.keras.models import Model                          # Model sınıfı
from tensorflow.keras.optimizers import Adam                       # Optimizasyon algoritması
from tensorflow.keras.callbacks import (                           # Eğitim kontrolcüleri
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau
)

import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================================================
# KONFİGÜRASYON VE SABİTLER
# =========================================================
DATA_DIR ="./chest_xray"   # Veri setinin ana dizini
IMG_SIZE = (224, 224)      # DenseNet121 için önerilen giriş boyutu
BATCH_SIZE = 64            # Her iterasyonda işlenecek görüntü sayısı
CLASS_MODE = "binary"      # İki sınıfımız var: PNEUMONIA vs NORMAL
EPOCHS = 20                # Maksimum eğitim tur sayısı

# =========================================================
# VERİ HAZIRLAMA VE ARTIRMA (DATA AUGMENTATION)
# =========================================================

# Eğitim verisi için jeneratör:
# Modelin ezberlemesini (overfitting) önlemek için görüntüler rastgele değiştirilir.
train_datagen = ImageDataGenerator(
    rescale=1./255,                 # Piksel değerlerini 0-1 arasına normalize et
    horizontal_flip=True,           # Rastgele yatay çevir
    rotation_range=10,              # Rastgele 10 derece döndür
    brightness_range=[0.8, 1.2],    # Parlaklığı %20 oranında değiştir
    validation_split=0.1            # Verinin %10'unu doğrulama (validation) için ayır
)

# Test verisi için jeneratör:
# Test verisinde ASLA artırma yapılmaz, sadece normalize edilir.
test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim setini yükle (subset='training')
train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    subset="training",
    shuffle=True  # Eğitim sırasında veri karıştırılır
)

# Doğrulama setini yükle (subset='validation')
# Eğitimden ayrılan %10'luk kısım
val_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    subset="validation",
    shuffle=False # Doğrulamada karıştırmaya gerek yok
)

# Test setini yükle (Ayrı 'test' klasöründen)
test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    shuffle=False # Test sırasında sıra bozulmamalı (confusion matrix için önemli)
)

# Sınıf isimlerini al ve örnek bir batch görselleştir
class_names = list(train_gen.class_indices.keys())
print(f"Sınıflar: {class_names}")

# Veri setinden rastgele bir batch çekip ilk 4 resmi gösterelim
images, labels = next(train_gen)

plt.figure(figsize=(10,4))
for i in range(4):
    ax = plt.subplot(1, 4, i+1)
    ax.imshow(images[i])
    # Label binary (0 veya 1) olduğu için int'e çevirip sınıf ismini yazdırıyoruz
    ax.set_title(class_names[int(labels[i])])
    ax.axis("off")

plt.tight_layout()
plt.show()

# =========================================================
# MODEL MİMARİSİ (TRANSFER LEARNING)
# =========================================================

# 1. Base Model (Temel Model) Yükleme:
# DenseNet121, ImageNet üzerinde eğitilmiş güçlü bir modeldir.
base_model = DenseNet121(
    weights="imagenet",       # Önceden eğitilmiş ağırlıkları kullan
    include_top=False,        # Son sınıflandırma katmanını dahil etme (biz ekleyeceğiz)
    input_shape=(*IMG_SIZE, 3)
)

# 2. Base Modelin Dondurulması:
# Mevcut ağırlıkların eğitim sırasında bozulmaması için donduruyoruz.
# Sadece kendi eklediğimiz katmanlar eğitilecek.
base_model.trainable = False

# 3. Özel Sınıflandırma Katmanlarının Eklenmesi:
x = base_model.output

# Özellik haritalarını vektöre çevir (Flatten yerine GAP daha verimlidir)
x = GlobalAveragePooling2D()(x)

# Tam bağlantılı katman (Özellikleri öğrenir)
x = Dense(128, activation="relu")(x)

# Dropout: Nöronların %50'sini rastgele kapatarak aşırı öğrenmeyi engeller
x = Dropout(0.5)(x)

# Çıkış Katmanı: Binary sınıflandırma (0 veya 1) için 'sigmoid' kullanılır.
# Eğer çoklu sınıf olsaydı 'softmax' kullanılırdı.
predictions = Dense(1, activation="sigmoid")(x)

# Modeli birleştir
model = Model(inputs=base_model.input, outputs=predictions) 

# =========================================================
# MODELİN DERLENMESİ VE CALLBACKLER
# =========================================================

model.compile(
    optimizer=Adam(learning_rate=1e-4), # Düşük öğrenme oranı (Transfer learning için ideal)
    loss="binary_crossentropy",         # İki sınıflı problem için kayıp fonksiyonu
    metrics=["accuracy"]
)

# Callbacks (Yardımcı Fonksiyonlar):
callbacks = [
    # Validation loss 3 epoch boyunca iyileşmezse eğitimi durdur
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    
    # Validation loss 2 epoch iyileşmezse öğrenme hızını (LR) düşür (0.2 ile çarp)
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    
    # En iyi val_loss değerine sahip modeli kaydet
    ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True),
]

print("Model Özeti:")
print(model.summary())

# =========================================================
# EĞİTİM (TRAINING)
# =========================================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# =========================================================
# TEST VE DEĞERLENDİRME
# =========================================================

print("Test seti üzerinde tahmin yapılıyor...")
# Modelin test seti üzerindeki olasılık tahminleri (0 ile 1 arası değerler)
pred_props = model.predict(test_gen, verbose=1)

# Olasılıkları etikete çevirme: 0.5'ten büyükse 1 (Pneumonia), küçükse 0 (Normal)
pred_labels = (pred_props >= 0.5).astype(int).ravel()

# Gerçek etiketleri al
true_labels = test_gen.classes

# Confusion Matrix (Karmaşıklık Matrisi) Oluşturma
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

# Matrisi görselleştir
plt.figure(figsize=(8,8))
disp.plot(cmap="Blues", ax=plt.gca()) # ax=plt.gca() mevcut figure üzerine çizer
plt.title("Confusion Matrix - Test Seti Performansı")
plt.show()