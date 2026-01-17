# 👁️ YOLOv8 ile Nesne Takibi ve Sayma Sistemleri

Bu depo (repository), **YOLOv8**, **OpenCV** ve **Python** kullanılarak geliştirilmiş gerçek zamanlı nesne tespiti, takibi (tracking) ve sayımı projelerini içerir. 

Proje kapsamında iki temel bilgisayarlı görü (computer vision) uygulaması geliştirilmiştir:
1. **Trafik Analizi:** Araç sınıflandırma ve şerit sayımı.
2. **Kalabalık Analizi:** İnsan giriş-çıkış (yönlü) sayımı.

---

## 📂 Proje İçeriği

### 1. 🚗 Trafik ve Araç Sayımı (`main_car.py`)
Belirlenen sanal bir çizgiyi geçen araçları tespit eder, takip eder ve sınıfına göre (Araba, Kamyon, Otobüs vb.) sayar.

* **Teknoloji:** YOLOv8 Tracking + Vektörel Geometri.
* **Yöntem:** `Cross Product` (Vektörel Çarpım) yöntemi ile aracın çizginin hangi tarafında olduğu hesaplanır.
* **Özellikler:**
    * Çoklu sınıf ayrımı (Car, Truck, Bus, Motorcycle).
    * Mükerrer sayımı önleyen ID takibi (ByteTrack).
    * Görselleştirilmiş takip çizgileri.

### 2. 🚶 İnsan Giriş-Çıkış Sayımı (`main_people.py`)
Kamera görüntüsünü dikey bir çizgi ile ikiye bölerek insanların sağa (Giren) veya sola (Çıkan) geçişlerini analiz eder.

* **Teknoloji:** YOLOv8 Tracking + Koordinat Delta Analizi.
* **Yöntem:** Nesnenin bir önceki karedeki (frame) X konumu ile şimdiki X konumu karşılaştırılarak hareket vektörü belirlenir.
* **Özellikler:**
    * Sadece "Person" sınıfı filtrelenir.
    * Giren ve Çıkan sayaçları ayrı ayrı tutulur.
    * Geçiş anında görsel uyarı (Renk değişimi).

---

