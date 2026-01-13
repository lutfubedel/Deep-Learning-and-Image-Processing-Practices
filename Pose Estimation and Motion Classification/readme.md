# 🏋️‍♂️ AI Destekli Squat Sayacı (AI Squat Counter)

Bu proje, **Python**, **OpenCV** ve **MediaPipe** kütüphanelerini kullanarak gerçek zamanlı görüntü işleme ile squat egzersizlerini takip eden, açıları hesaplayan ve tekrarları otomatik olarak sayan bir uygulamadır.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Estimation-orange)

## 📋 İçindekiler
- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Algoritma Mantığı](#algoritma-mantığı)
- [Gereksinimler](#gereksinimler)

## 🚀 Proje Hakkında
Bu uygulama, webcam veya video dosyası üzerinden insan vücudunu algılar ve **Kalça - Diz - Ayak Bileği** noktaları arasındaki açıyı trigonometrik olarak hesaplar. Bu açıya göre kişinin "Eğilme" (Squat Down) veya "Kalkma" (Squat Up) durumunda olduğunu belirleyerek tekrarları sayar.

## ✨ Özellikler
* **Gerçek Zamanlı İskelet Takibi:** MediaPipe Pose modeli ile yüksek doğrulukta vücut analizi.
* **Otomatik Açı Hesaplama:** NumPy kullanılarak eklem açılarının anlık hesaplanması.
* **Durum Analizi:** "Hazır", "Aşağı" ve "Yukarı" durumlarını algılama.
* **Görsel Arayüz:** Ekranda anlık tekrar sayısı, hareket durumu ve açı değerinin gösterimi.
* **Hata Toleransı:** Görüntüde insan algılanamadığında programın çökmesini engelleyen yapı.

## 🧠 Algoritma Mantığı
Sayacın çalışması için belirli açı eşikleri (thresholds) belirlenmiştir:
* Noktalar: Sol Kalça, Sol Diz ve Sol Ayak Bileği koordinatları alınır.
* Açı Hesabı: arctant2 fonksiyonu ile dizdeki açı hesaplanır.
* Squat Down (İniş): Eğer açı 90 derecenin altına düşerse, durum Asagi olarak işaretlenir.
* Squat Up (Kalkış): Eğer açı 160 derecenin üzerine çıkarsa ve önceki durum Asagi ise, hareket tamamlanmış sayılır ve sayaç 1 artar.

## 📦 Gereksinimler
* Python 3.x
* opencv-python
* mediapipe
* numpy
  
## 📸 Sonuçlar

Modelin test aşamasındaki performansı aşağıda gösterilmiştir.

<table>
  <tr>
    <td align="center" width="50%"><b>Orijinal Görüntü</b></td>
    <td align="center" width="50%"><b>Tespit Sonucu</b></td>
  </tr>
  <tr>
    <td><img src="images/img_1.png" width="100%"></td>
    <td><img src="images/imag_2.png" width="100%"></td>
  </tr>
</table>
