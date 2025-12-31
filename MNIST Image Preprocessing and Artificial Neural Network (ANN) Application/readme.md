# MNIST Görüntü Ön İşleme ve Yapay Sinir Ağı (ANN) Uygulaması

Bu proje, **MNIST el yazısı rakam veri seti** üzerinde **OpenCV tabanlı görüntü ön işleme teknikleri** uygulanarak elde edilen özellikler ile **Yapay Sinir Ağı (Artificial Neural Network - ANN)** eğitilmesini amaçlamaktadır.

Proje kapsamında klasik ham piksel kullanımı yerine, kenar tabanlı özellik çıkarımı yapılarak sınıflandırma performansı incelenmiştir.

---

## 🚀 Projenin Amacı

- Görüntü ön işleme tekniklerinin sınıflandırma üzerindeki etkisini incelemek  
- OpenCV kullanarak histogram eşitleme, bulanıklaştırma ve kenar algılama uygulamak  
- Ön işlenmiş görüntüler ile ANN modeli eğitmek  
- Eğitim ve doğrulama sonuçlarını görselleştirmek  

---

## 🧠 Kullanılan Teknolojiler

- **Python 3.9+**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**

---

## 📂 Veri Seti

- **MNIST Handwritten Digits Dataset**
- 28x28 boyutunda gri seviye el yazısı rakam görüntüleri
- 0–9 arası 10 sınıf

---

## 🔍 Görüntü Ön İşleme Adımları

Her bir MNIST görüntüsü için aşağıdaki işlemler uygulanmıştır:

1. **Histogram Equalization**  
   - Görüntü kontrastını artırmak için kullanılmıştır.

2. **Gaussian Blur**  
   - Gürültüyü azaltmak ve kenar algılamayı iyileştirmek için uygulanmıştır.

3. **Canny Edge Detection**  
   - Rakamların kenarlarını belirgin hale getirmek için kullanılmıştır.

4. **Flatten & Normalization**  
   - 28x28 görüntüler 784 boyutlu vektöre dönüştürülmüş ve 0–1 aralığında normalize edilmiştir.

---

## 🧪 Model Mimarisi (ANN)

Kullanılan yapay sinir ağı mimarisi:

- Girdi Katmanı: 784 nöron  
- Gizli Katman 1: 128 nöron (ReLU)  
- Dropout: %50  
- Gizli Katman 2: 64 nöron (ReLU)  
- Çıkış Katmanı: 10 nöron (Softmax)

**Kayıp Fonksiyonu:** Sparse Categorical Crossentropy  
**Optimizasyon:** Adam  
**Öğrenme Oranı:** 0.001  

---

## 📊 Eğitim Detayları

- Eğitim verisi: 10.000 örnek  
- Test verisi: 2.000 örnek  
- Epoch sayısı: 10  
- Batch size: 32  

Eğitim sürecinde **accuracy** ve **loss** değerleri hem eğitim hem doğrulama seti için izlenmiştir.

---

## 📈 Sonuçlar

- Model, ön işlenmiş kenar tabanlı özellikler ile makul bir doğruluk oranına ulaşmıştır.
- Ham piksel tabanlı yaklaşıma kıyasla farklı bir özellik çıkarım yöntemi denenmiştir.
- Eğitim süreci grafiklerle analiz edilmiştir.

---

## 📊 Çıktı
![Görsel](images/Figure_1.png)
