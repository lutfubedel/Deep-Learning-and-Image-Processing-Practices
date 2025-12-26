# =========================================================
# Gerekli kütüphanelerin yüklenmesi
# =========================================================
# python -m venv venv
# .\venv\Scripts\activate
# pip install tensorflow matplotlib opencv-python
# pip freeze > requirements.txt

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


# =========================================================
# MNIST veri setinin yüklenmesi
# =========================================================
# x_train, x_test: 28x28 boyutunda gri seviye görüntüler
# y_train, y_test: 0-9 arası sınıf etiketleri
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")


# =========================================================
# Görüntü ön işleme adımlarının görselleştirilmesi
# =========================================================
img = x_train[0]
stages = {"Orijinal Görüntü": img}

# Histogram Equalization (Kontrast artırma)
img_hist_eq = cv2.equalizeHist(img)
stages["Histogram Equalization"] = img_hist_eq

# Gaussian Blur (Gürültü azaltma)
img_gaussian = cv2.GaussianBlur(img_hist_eq, (5, 5), 0)
stages["Gaussian Blurring"] = img_gaussian

# Canny Edge Detection (Kenar algılama)
img_canny = cv2.Canny(img_gaussian, 50, 100)
stages["Canny Edge Detection"] = img_canny


# Ön işleme adımlarının ekranda gösterilmesi
fig, axes = plt.subplots(2, 2, figsize=(6, 6))
axes = axes.flat

for ax, (title, im) in zip(axes, stages.items()):
    ax.imshow(im, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.suptitle("Görüntü Ön İşleme Aşamaları")
plt.tight_layout()
plt.show()


# =========================================================
# Ön işleme fonksiyonu
# =========================================================
def preprocess_images(image):
    """
    Tek bir MNIST görüntüsü için:
    - Histogram Equalization
    - Gaussian Blur
    - Canny Edge Detection
    - Normalize ve flatten işlemleri
    """
    img_hist_eq = cv2.equalizeHist(image)
    img_gaussian = cv2.GaussianBlur(img_hist_eq, (5, 5), 0)
    img_canny = cv2.Canny(img_gaussian, 50, 150)

    # 28x28 görüntüyü 1D vektöre çevir ve normalize et
    features = img_canny.flatten() / 255.0
    return features


# =========================================================
# Eğitim ve test veri setlerinin hazırlanması
# =========================================================
num_train = 10000
num_test = 2000

X_train = np.array([preprocess_images(img) for img in x_train[:num_train]])
X_test = np.array([preprocess_images(img) for img in x_test[:num_test]])

y_train_subset = y_train[:num_train]
y_test_subset = y_test[:num_test]


# =========================================================
# Yapay Sinir Ağı (ANN) modelinin oluşturulması
# =========================================================
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])


# =========================================================
# Modelin derlenmesi
# =========================================================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())


# =========================================================
# Modelin eğitilmesi
# =========================================================
history = model.fit(
    X_train,
    y_train_subset,
    validation_data=(X_test, y_test_subset),
    epochs=10,
    batch_size=32,
    verbose=2
)


# =========================================================
# Modelin test edilmesi
# =========================================================
test_loss, test_accuracy = model.evaluate(X_test, y_test_subset)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


# =========================================================
# Eğitim sürecinin görselleştirilmesi
# =========================================================
plt.figure(figsize=(12, 5))

# Accuracy grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Accuracy')
plt.plot(history.history['val_accuracy'], label='Doğrulama Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Loss')
plt.plot(history.history['val_loss'], label='Doğrulama Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
