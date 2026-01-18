# Deep Learning & Image Processing Practices

Bu depo, Derin Öğrenme ve Bilgisayarlı Görü alanlarında geliştirdiğim, gerçek hayat senaryolarına ve akademik çalışmalara odaklanan projeleri içerir. Her bir proje, modern algoritmaların pratik uygulamalarını ve performans analizlerini barındırır.

## 🚀 Proje Kataloğu

Aşağıdaki tabloda, bu repo içerisinde yer alan projelerin özetlerini, kullanılan teknolojileri ve ilgili kodlara erişim linklerini bulabilirsiniz.

| Proje İsmi | Açıklama & Amaç | Kullanılan Teknolojiler | Kaynak Kod |
| :--- | :--- | :--- | :---: |
| **Image Captioning** | Görselleri analiz ederek içeriği tanımlayan anlamlı metinler (altyazı) üretir. Vision Encoder-Decoder mimarileri üzerine kuruludur. | ![Python](https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![HuggingFace](https://img.shields.io/badge/Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black) | [📂 İncele](./Image_Captioning) |
| **Neural Style Transfer** | Bir referans görselin sanatsal stilini (örn. Van Gogh) koruyarak, içerik görseline aktarır. VGG19 tabanlı özellik çıkarımı kullanır. | ![Python](https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | [📂 İncele](./Neural_Style_Transfer) |
| **YOLO Object Detection** | Gerçek zamanlı video akışlarında nesne tespiti, takibi ve sayımı (araç, yaya vb.) yapar. Yüksek FPS ve doğruluk odaklıdır. | ![Python](https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=white) ![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=flat&logo=yolo&logoColor=black) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) | [📂 İncele](./YOLO_Projects) |
| **Pose Estimation** | İnsan vücut iskeletini (landmark) tespit ederek hareket analizi yapar. Squat sayma veya duruş bozukluğu tespiti gibi uygulamalar içerir. | ![Python](https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=white) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0099CC?style=flat&logo=google&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) | [📂 İncele](./Pose_Estimation) |

---

## 🛠 Genel Kurulum

Projeleri yerel ortamınızda çalıştırmak için aşağıdaki genel adımları takip edebilirsiniz. *Her projenin kendi klasöründe daha spesifik `requirements.txt` dosyaları bulunabilir.*

```bash
# Repoyu klonlayın
git clone [https://github.com/lutfubedel/Deep-Learning-and-Image-Processing-Practices.git](https://github.com/lutfubedel/Deep-Learning-and-Image-Processing-Practices.git)

# Proje dizinine girin
cd Deep-Learning-and-Image-Processing-Practices

# Sanal ortam oluşturun ve aktif edin
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Temel bağımlılıkları yükleyin
pip install torch torchvision opencv-python transformers numpy matplotlib
