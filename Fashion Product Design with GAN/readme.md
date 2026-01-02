# CNN ile Çiçek Türü Sınıflandırma

Bu proje, **TensorFlow ve Keras** kullanılarak **Convolutional Neural Network (CNN)** mimarisi ile çiçek görüntülerinin sınıflandırılmasını amaçlamaktadır.  
Model, **TF Flowers** veri seti üzerinde eğitilmiştir.

---

## 🎯 Amaç

- CNN tabanlı bir görüntü sınıflandırma modeli geliştirmek
- Data augmentation ile modelin genelleme yeteneğini artırmak
- Eğitim ve doğrulama performanslarını görsel olarak analiz etmek

---

## 🧠 Kullanılan Teknolojiler

- Python
- TensorFlow & Keras
- TensorFlow Datasets
- Matplotlib

---

## 📂 Veri Seti

**TF Flowers** veri seti kullanılmıştır.

**Sınıflar:**
- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

**Bölünme:**
- %80 Eğitim
- %20 Doğrulama

---

## 🏗️ Model Özeti

- 3 adet Convolution + MaxPooling bloğu  
- 1 adet Dense katman  
- Dropout ile overfitting önleme  
- Softmax çıkış katmanı (5 sınıf)

---

## 🔄 Data Augmentation

Eğitim sırasında:
- Yatay çevirme
- Parlaklık ve kontrast değişimi
- Rastgele kırpma  

uygulanmıştır.

---

## ⚙️ Eğitim

- Optimizer: Adam  
- Loss: Sparse Categorical Crossentropy  
- Callback’ler:
  - EarlyStopping
  - ReduceLROnPlateau
  - ModelCheckpoint

---
## 📊 Çıktı
![Görsel](images/img-1.png)
