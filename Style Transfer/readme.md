# PyTorch ile Sinirsel Stil Aktarımı (Neural Style Transfer)

Bu proje, PyTorch kullanarak **Neural Style Transfer (NST)** algoritmasının bir uygulamasını içerir. Leon A. Gatys ve arkadaşlarının geliştirdiği bu teknik, bir görüntünün (içerik) ana hatlarını korurken, başka bir görüntünün (stil) sanatsal dokusunu ve renklerini ona aktararak yeni bir sanat eseri oluşturur.

## 🚀 Özellikler

* **VGG19 Mimarisi:** Özellik çıkarımı (feature extraction) için önceden eğitilmiş VGG19 ağı kullanılır.
* **Gram Matrisi:** Stil temsili için Gram Matrisleri hesaplanır.
* **Özelleştirilebilir Ağırlıklar:** Stil ve içerik dengesi (alpha/beta oranı) ayarlanabilir.
* **CUDA Desteği:** GPU varsa otomatik algılar ve işlemi hızlandırır.
* **İlerleme Çubuğu:** `tqdm` kütüphanesi ile eğitim adımları görsel olarak takip edilebilir.


## 📸 Sonuçlar

Modelin örnek çıktıları aşağıda verilmiştir

<table>
  <tr>
    <td align="center" width="50%"><b>Orijinal Görüntü</b></td>
    <td align="center" width="50%"><b>Style</b></td>
    <td align="center" width="50%"><b>Sonuç</b></td>
  </tr>
  <tr>
    <td><img src="content.jpeg" width="100%"></td>
    <td><img src="style.jpg" width="100%"></td>
    <td><img src="Figure_1.png" width="100%"></td>
    
  </tr>
</table>


