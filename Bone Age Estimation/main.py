# ==========================================
# KURULUM VE HAZIRLIK KOMUTLARI (Terminal)
# ==========================================
# python -m venv venv
# .\venv\Scripts\activate
# pip install tensorflow pandas matplotlib opencv-python scikit-learn
# # Kaynak Veri Seti: https://www.kaggle.com/datasets/kmader/rsna-bone-age
# pip freeze > requirements.txt

# Gerekli kütüphanelerin içe aktarılması
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import MeanAbsoluteError

# ==========================================
# VERI SETI YUKLEME VE PREPROCESSING
# ==========================================

# Veri setini (CSV) yükle
df = pd.read_csv("boneage-training-dataset.csv")
print(df.head())

image_folder = "boneage-training-dataset"

# Klasördeki mevcut resim dosyalarını kontrol et ve olmayanları veri setinden çıkar
available_files = set(os.listdir(image_folder))
available_ids = set(f.replace(".png", "") for f in available_files if f.endswith(".png"))
df = df[df["id"].astype(str).isin(available_ids)].reset_index(drop=True)    
print(f"Toplam {len(df)} adet resim bulunuyor.")

# Kemik yaşını normalize et (Maksimum yaş 240 ay kabul edilerek 0-1 arasına getirilir)
df["boneage"] = df["boneage"] / 240
# Resim dosya yollarını oluştur ve dataframe'e ekle
df["path"] = df["id"].apply(lambda x: os.path.join(image_folder, f"{x}.png"))
print(df.head()) 

# Veri setindeki kemik yaşı dağılımını görselleştir
plt.hist(df["boneage"], bins=50)
plt.xlabel("Kemik Yaşı")
plt.ylabel("Frekans")
plt.title("Kemik Yaşı Dağılımı")
plt.tight_layout()
plt.show()  

# Resimleri yükleyen ve işleyen fonksiyon
def load_images(df, img_size=128):
    images = []
    valid_indices = []
    for i, path in enumerate(df["path"]):
        # Resmi gri tonlamalı olarak oku
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Resmi belirtilen boyuta yeniden boyutlandır
        img = cv2.resize(img, (img_size, img_size))
        # Piksel değerlerini 0-1 arasına normalize et
        img = img / 255.0
        images.append(img)
        valid_indices.append(i)
    
    # Sadece başarıyla yüklenen resimlerin etiketlerini al
    new_df = df.iloc[valid_indices].reset_index(drop=True)
    # Resim verisini modelin beklediği formata (N, 128, 128, 1) dönüştür
    return np.array(images).reshape(-1, img_size, img_size, 1), new_df["boneage"].values


# Resimleri belleğe yükle
x, y = load_images(df)
print(f"Yüklenen veri boyutu: {x.shape}")

# Veriyi eğitim ve test setlerine ayır (%85 eğitim, %15 test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
print(f"Eğitim ve test boyutları: {x_train.shape}, {x_test.shape}, {y_train.shape}, {y_test.shape}")    

# Veri çoğaltma (Data Augmentation) nesnesini oluştur
# Bu, eğitim verisini çeşitlendirerek modelin genelleme yeteneğini artırır
datagen = ImageDataGenerator(
    width_shift_range=0.1,  # Yatay kaydırma
    height_shift_range=0.1, # Dikey kaydırma
    zoom_range=0.1,         # Yakınlaştırma/Uzaklaştırma
    horizontal_flip=True,   # Yatay çevirme
)

datagen.fit(x_train)

# CNN Modelini oluştur
model = Sequential()

# İlk Konvolüsyon Katmanı: 32 filtre, 3x3 çekirdek boyutu
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation="relu"))
model.add(MaxPooling2D((2, 2))) # Boyut azaltma

# İkinci Konvolüsyon Katmanı: 64 filtre
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))

# Düzleştirme (Flatten) ve Tam Bağlantılı (Dense) Katmanlar
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5)) # Aşırı öğrenmeyi (overfitting) önlemek için dropout
model.add(Dense(1, activation="linear")) # Çıkış katmanı (Regresyon için linear aktivasyon)

# Modeli derle
# Optimizer: Adam, Kayıp Fonksiyonu: Ortalama Mutlak Hata (MAE)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mae",
    metrics=[MeanAbsoluteError()]
)

# Callbacks (Geri Çağrılar) tanımla
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True), # İyileşme durursa eğitimi erken bitir
    ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True), # En iyi modeli kaydet
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5) # Plato durumunda öğrenme hızını azalt
]   

# Modeli eğit
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32), # Artırılmış veri ile eğitim
    epochs=20, # Eğitim döngüsü sayısı (Epoch)
    validation_data=(x_test, y_test), # Doğrulama verisi
    callbacks=callbacks
)   

model.summary() 

# Eğitim performansını görselleştir
plt.plot(history.history["loss"], label="Eğitim MAE")
plt.plot(history.history["val_loss"], label="Doğrulama MAE")
plt.ylabel("Ortalama Mutlak Hata (MAE)")
plt.xlabel("Epoch")
plt.title("Eğitim Performansı")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Test seti üzerinde tahminler yap ve sonuçları görselleştir
preds = model.predict(x_test) * 240 # Tahminleri gerçek yaşa dönüştür
actual = y_test * 240 # Gerçek değerleri normale döndür

plt.figure()
for i in range(10): # İlk 10 örneği göster
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(128, 128), cmap="gray")
    plt.title(f"Gerçek: {actual[i]:.1f}\nTahmin: {preds[i][0]:.1f}")
    plt.axis("off")
plt.tight_layout()
plt.show()