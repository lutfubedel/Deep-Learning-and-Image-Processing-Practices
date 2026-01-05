# ==========================================
# KURULUM VE HAZIRLIK KOMUTLARI (Terminal)
# ==========================================
# python -m venv venv
# .\venv\Scripts\activate
# pip install tensorflow opencv-python numpy matplotlib scikit-learn pandas
# pip freeze > requirements.txt
# Veri seti linki: https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# ==========================================
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# ==========================================
def load_dataset(root, img_size=(128,128)):
    """
    Belirtilen kök dizinden görüntüleri ve maskeleri okur,
    yeniden boyutlandırır ve normalize eder.
    """
    images, maskes = [], [] 
    
    # Klasör yapısında her bir 'tile' (karo/bölge) içinde geziniyoruz
    for tile in sorted(os.listdir(root)):
        img_dir = os.path.join(root, tile, "images")
        mask_dir = os.path.join(root, tile, "masks")
        
        # Eğer klasörler yoksa o tile'ı atla
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            continue

        for f in os.listdir(img_dir):
            if not f.lower().endswith('.jpg'): continue
            
            img_path = os.path.join(img_dir, f)
            # Maske ismi genelde görüntü ismiyle aynıdır, sadece uzantısı farklı olabilir
            mask_path = os.path.join(mask_dir, os.path.splitext(f)[0] + ".png")
            
            if not os.path.exists(mask_path): continue

            # --- Görüntü İşleme ---
            # OpenCV BGR okur, biz RGB'ye çeviriyoruz
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            # Boyutlandırma ve Normalizasyon (0-255 aralığını 0-1 aralığına çeker)
            img = cv2.resize(img, img_size) / 255.0

            # --- Maske İşleme ---
            # Maskeleri gri tonlamalı (tek kanallı) okuyoruz
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            # Model çıkışına uygun olması için boyut genişletme: (128, 128) -> (128, 128, 1)
            mask = np.expand_dims(mask, axis=-1) / 255.0

            images.append(img)
            maskes.append(mask)

    # Listeleri Numpy dizilerine çeviriyoruz (Model beslemesi için gereklidir)
    return np.array(images, dtype="float32"), np.array(maskes, dtype="float32")

# Veri setini yükle
# NOT: "aeiral_dataset" klasörünün proje dizininde olduğundan emin olun.
x, y = load_dataset("aeiral_dataset", img_size=(128,128))
print("Toplam Örnek Sayısı:", len(x))

# Veriyi Eğitim (%80) ve Doğrulama (%20) olarak ayır
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42) 
print("Eğitim Örnek Sayısı:", len(x_train))
print("Doğrulama Örnek Sayısı:", len(x_val))


# ==========================================
# 2. MODEL MİMARİSİ (U-NET)
# ==========================================
def unet_model(input_size=(128, 128, 3)):
    inputs = keras.Input(input_size)

    # --- ENCODER (Aşağı İnen Yol - Contracting Path) ---
    # Görüntüdeki özellikleri (kenar, doku, şekil) çıkarmak için kullanılır.
    # Her adımda görüntü boyutu yarıya düşer (MaxPooling), filtre sayısı artar.

    # Blok 1
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1) # Boyutu yarıya indirir

    # Blok 2
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    # Blok 3
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)  

    # Blok 4
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D()(c4)

    # --- BOTTLENECK (Dar Boğaz) ---
    # En derin özelliklerin çıkarıldığı yer.
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)  

    # --- DECODER (Yukarı Çıkan Yol - Expansive Path) ---
    # Görüntüyü orijinal boyutuna geri getirir ve maskeyi oluşturur.
    # 'concatenate' (Skip Connection): Encoder'daki kaybolan mekansal bilgiyi geri kazanmak için kullanılır.

    # Blok 6 (Yukarı Örnekleme)
    u6 = layers.Conv2DTranspose(128, 2, strides=(2,2), padding='same')(c5) # Boyutu 2 katına çıkarır
    u6 = layers.concatenate([u6, c4]) # Skip connection: c4 bloğundaki detayları ekle
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)   

    # Blok 7
    u7 = layers.Conv2DTranspose(64, 2, strides=(2,2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3]) # Skip connection: c3 ile birleştir
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)

    # Blok 8
    u8 = layers.Conv2DTranspose(32, 2, strides=(2,2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2]) # Skip connection: c2 ile birleştir
    c8 = layers.Conv2D(32, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(32, 3, activation='relu', padding='same')(c8)    

    # Blok 9
    u9 = layers.Conv2DTranspose(16, 2, strides=(2,2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1]) # Skip connection: c1 ile birleştir (orijinal boyut detayları)
    c9 = layers.Conv2D(16, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(16, 3, activation='relu', padding='same')(c9)

    # --- ÇIKIŞ KATMANI ---
    # 1 filtreli Conv2D ve 'sigmoid' aktivasyonu. 
    # Çünkü her piksel için 0 (arka plan) veya 1 (hedef) olasılığı istiyoruz.
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9) 
    
    model = keras.Model(inputs, outputs)
    return model    


# Modeli oluştur
unet_model = unet_model()

# Modeli derle: Binary Crossentropy (İkili sınıflandırma kaybı) kullanılır.
unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==========================================
# 3. EĞİTİM (TRAINING)
# ==========================================
callbacks = [
    # En iyi modeli kaydet (val_loss düştükçe)
    keras.callbacks.ModelCheckpoint("model_best.h5", save_best_only=True),
    # Öğrenme hızı tıkandığında (loss düşmüyorsa) öğrenme oranını azalt
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    # Model gelişmeyi durdurursa eğitimi erken bitir
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
]

history = unet_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=16,
    callbacks=callbacks
)

# Eğitim kaybı grafiği
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Eğitim Kaybı (Train Loss)')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (Val Loss)')
plt.title("Eğitim Sürecindeki Kayıp Değişimi")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ==========================================
# 4. TAHMİN VE GÖRSELLEŞTİRME
# ==========================================
def show_predictions(idx=0):
    """
    Doğrulama setinden rastgele bir görüntüyü alır, modele tahmin ettirir
    ve Girdi - Gerçek Maske - Tahmin Edilen Maske şeklinde gösterir.
    """
    img = x_val[idx]
    mask_true = y_val[idx].squeeze() # Boyut azaltma: (128,128,1) -> (128,128)
    
    # Model tahmini (4 boyutlu tensor ister: batch_size, h, w, c)
    mask_raw = unet_model.predict(img[None, ...])[0].squeeze()
    
    # Eşikleme (Thresholding): 0.5'ten büyükse 1 (beyaz), değilse 0 (siyah) yap
    mask_pred = (mask_raw > 0.5).astype(np.float32)

    plt.figure(figsize=(12, 4))
    
    # Orijinal Görüntü
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Girdi Görüntüsü (Input)")
    plt.axis("off")

    # Gerçek Maske (Etiket)
    plt.subplot(1, 3, 2)
    plt.imshow(mask_true, cmap='gray')
    plt.title("Gerçek Maske (Ground Truth)")
    plt.axis("off")

    # Modelin Tahmini
    plt.subplot(1, 3, 3)
    plt.title("Tahmin Edilen Maske")
    plt.imshow(mask_pred, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Örnek bir tahmin göster
show_predictions(1)