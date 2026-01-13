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

## 🧠 Nasıl Çalışır?
Bu proje, derin öğrenme (deep learning) sınıflandırması yerine geometrik öklid mesafesi mantığıyla çalışır. MediaPipe Face Mesh yüzdeki 468 noktayı (landmark) bize verir.

Biz şu noktaları kullanıyoruz:

* Şaşkınlık (Surprised): Kaş ortası ile göz bebeği arasındaki dikey mesafe ölçülür. Mesafe belirli bir eşik değerini (Threshold) geçerse kişi "Şaşkın" kabul edilir.
* Mutluluk (Happy): Ağzın sağ ve sol köşeleri arasındaki yatay mesafe ölçülür. Gülümseme sırasında bu mesafe arttığı için eşik değeri geçildiğinde "Mutlu" kabul edilir.
* Nötr (Neutral): Yukarıdaki koşullar sağlanmıyorsa ifade "Nötr"dür.

Not: Eşik değerleri (25 ve 60 piksel) kameraya olan uzaklığa göre kod içerisinden optimize edilebilir.
