# 🎭 MediaPipe ile Gerçek Zamanlı Duygu Analizi

Bu proje, **Python**, **OpenCV** ve **MediaPipe** kütüphanelerini kullanarak web kamerası üzerinden gerçek zamanlı yüz takibi (Face Mesh) yapar ve basit geometrik hesaplamalarla temel duyguları (Mutlu, Şaşkın, Nötr) tespit eder.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange)

## 🚀 Özellikler

* **Gerçek Zamanlı Takip:** Web kamerasından alınan görüntüyü gecikmesiz işler.
* **Yüz Ağı (Face Mesh):** Yüz üzerindeki 468 farklı noktayı tespit eder ve çizer.
* **Geometrik Analiz:** Yapay zeka eğitimi gerektirmeden, yüz noktaları arasındaki mesafeyi ölçerek anlık duygu tahmini yapar.
* **Hafif ve Hızlı:** Düşük donanımlarda bile yüksek performansla çalışır.

## 🛠️ Gereksinimler

Projeyi çalıştırmadan önce bilgisayarınızda Python'un kurulu olması gerekir. Kullanılan kütüphaneler:

* `opencv-python` (Görüntü işleme)
* `mediapipe` (Yüz tespiti)
* `numpy` (Matematiksel hesaplamalar)

## 📦 Kurulum

Projeyi bilgisayarınıza kurmak için aşağıdaki adımları izleyin:

1.  **Projeyi Klonlayın veya İndirin:**
    ```bash
    git clone [https://github.com/kullaniciadi/duygu-analizi.git](https://github.com/kullaniciadi/duygu-analizi.git)
    cd duygu-analizi
    ```

2.  **Sanal Ortam Oluşturun (Önerilen):**
    ```bash
    # Windows için:
    python -m venv venv
    .\venv\Scripts\activate

    # Mac/Linux için:
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install mediapipe opencv-python numpy
    ```

## ▶️ Kullanım

Kurulum tamamlandıktan sonra terminal üzerinden scripti çalıştırabilirsiniz:

```bash
python main.py
