# 🖊️ Gerçek Zamanlı El Yazısı Rakam Tanıma (Real-Time Digit Recognition)

Bu proje, TensorFlow ve Keras kullanılarak eğitilmiş bir Derin Öğrenme (CNN) modelini, OpenCV aracılığıyla gerçek zamanlı kamera görüntüsü üzerinde çalıştırır. El yazısı rakamları (0-9) webcam üzerinden anlık olarak tespit eder ve sınıflandırır.

## 🚀 Özellikler

* **Derin Öğrenme:** MNIST veri seti üzerinde eğitilmiş, %98+ başarımlı Convolutional Neural Network (CNN) modeli.
* **Veri Zenginleştirme (Data Augmentation):** Modelin ezberlemesini önlemek için döndürme, yakınlaştırma ve kaydırma işlemleri.
* **Görüntü İşleme:** OpenCV kullanılarak görüntü gri tonlamaya çevrilir, gürültüden arındırılır ve eşikleme (thresholding) uygulanır.
* **Gerçek Zamanlı Tespit:** Webcam üzerinden alınan görüntüyü anlık olarak işler ve tahmin sonucunu güven oranıyla birlikte ekrana yansıtır.

## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

### Gereksinimler

* Python 3.7+
* Sanal Ortam (Önerilir)

### Adım Adım Kurulum

1.  Proje dosyasını klonlayın veya indirin.
2.  Bir sanal ortam oluşturun ve aktif edin:
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install opencv-python tensorflow matplotlib numpy
    ```

## 📂 Dosya Yapısı

* `cnn.py`: MNIST veri setini indirir, CNN modelini kurar, eğitir ve `.h5` dosyası olarak kaydeder.
* `predict_from_camera.py`: Webcam'i açar, kaydedilen modeli yükler ve gerçek zamanlı tahmin yapar.
* `mnist_cnn_model.h5`: Eğitilmiş model dosyası (Eğitim sonrası oluşur).
