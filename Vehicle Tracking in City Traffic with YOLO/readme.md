# 🚗 YOLOv8 ile Araç Tespit ve Takip Sistemi

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Latest-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

Bu proje, **Ultralytics YOLOv8** ve **OpenCV** kütüphanelerini kullanarak video üzerindeki araçları (veya diğer nesneleri) tespit eder ve takip eder (Object Tracking). `ByteTrack` algoritması sayesinde nesnelere benzersiz bir ID atanır ve nesneler kareler boyunca izlenir.

## 🌟 Özellikler

* **Gerçek Zamanlı Tespit:** YOLOv8 Nano modeli ile hızlı tespit.
* **Nesne Takibi (Tracking):** `persist=True` parametresi ile nesne kimliklerinin (ID) korunması.
* **Video Kaydı:** İşlenen görüntülerin `.avi` formatında dışa aktarılması.
* **Görselleştirme:** Tespit edilen nesnelerin etrafına kutu (bounding box), güven skoru ve sınıf isminin çizilmesi.

## 📂 Proje Yapısı

```text
├── videos/                  # İşlenecek kaynak videolar buraya eklenir
│   └── IMG_5268.MOV
├── main.py                  # Ana çalışma dosyası
├── requirements.txt         # Gerekli kütüphaneler
├── yolov8n.pt               # İlk çalıştırmada otomatik inen model dosyası
├── output_video.avi         # Çıktı dosyası (Script çalıştıktan sonra oluşur)
└── README.md                # Proje dokümantasyonu****
```

## 📸 Sonuçlar

Modelin test aşamasındaki performansı aşağıda gösterilmiştir.

<table>
  <tr>
    <td align="center" width="50%"><b>Orijinal Görüntü</b></td>
    <td align="center" width="50%"><b>Tespit Sonucu</b></td>
  </tr>
  <tr>
    <td><img src="images/test_1.jpg" width="100%"></td>
    <td><img src="images/test_1_detections.jpg" width="100%"></td>
  </tr>
</table>
