# Deep Learning & Image Processing Practices

Bu depo, Derin Öğrenme ve Bilgisayarlı Görü alanlarında geliştirdiğim, gerçek hayat senaryolarına ve akademik çalışmalara odaklanan projeleri içerir. Her bir proje, modern algoritmaların pratik uygulamalarını ve performans analizlerini barındırır.

## 🚀 Proje Kataloğu

| # | Proje İsmi | Açıklama & Amaç | Teknoloji Yığını | Kod |
|:-:| :--- | :--- | :--- | :---: |
| **1** | **RSNA Kemik Yaşı Tahmini** | El röntgen görüntülerini kullanarak çocukların kemik yaşını (ay cinsinden) tahmin eden derin öğrenme tabanlı bir Regresyon modelidir (CNN) | ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) | [📂](https://github.com/lutfubedel/Deep-Learning-and-Image-Processing-Practices/tree/main/Bone%20Age%20Estimation) |
| **2** | **YOLOv8 ile Nesne Takibi ve Sayma Sistemleri** | Proje kapsamında iki temel bilgisayarlı görü (computer vision) uygulaması geliştirilmiştir: 1. **Trafik Analizi:** Araç sınıflandırma ve şerit sayımı. 2. **Kalabalık Analizi:** İnsan giriş-çıkış (yönlü) sayımı. | ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![YOLOv8](https://img.shields.io/badge/YOLO-v8-green) ![OpenCV](https://img.shields.io/badge/OpenCV-Latest-red) | [📂](https://github.com/lutfubedel/Deep-Learning-and-Image-Processing-Practices/tree/main/Counting%20Vehicles%20and%20People%20with%20YOLO%20via%20Video) |
| **3** | **Resimden Metin Üretme** | Deep Learning ve Hugging Face Transformers kütüphanesini kullanarak görselleri analiz eder ve içerikleri hakkında İngilizce açıklamalar (caption) üretir. | ![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-11557c) ![Requests](https://img.shields.io/badge/Requests-HTTP-black) | [📂](https://github.com/lutfubedel/Deep-Learning-and-Image-Processing-Practices/tree/main/Creating%20a%20Description%20from%20an%20Image) |
| **4** | **Gerçek Zamanlı El Yazısı Rakam Tanıma** |  yazısı rakamları (0-9) webcam üzerinden anlık olarak tespit eder ve sınıflandırır. | ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white) ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | [📂 Git](./Lane_Tracking) |
| **5** | **DCGAN ile Görüntü Üretimi** | Model, Fashion MNIST veri setindeki kıyafet görüntülerini öğrenerek, rastgele gürültüden (noise) tamamen yeni ve yapay kıyafet tasarımları üretir. | ![MediaPipe](https://img.shields.io/badge/-MediaPipe-0099CC?logo=google&logoColor=white) ![CV2](https://img.shields.io/badge/-OpenCV-green?logo=opencv&logoColor=white) | [📂 Git](./Pose_Estimation) |
| **6** | **CNN ile Çiçek Türü Sınıflandırma** | Bu proje, TensorFlow ve Keras kullanılarak Convolutional Neural Network (CNN) mimarisi ile çiçek görüntülerinin sınıflandırılmasını amaçlamaktadır. | ![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white) ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) | [📂 Git](./Klasor_Adi) |
| **7** | **MNIST Görüntü Ön İşleme** | Bu proje, MNIST el yazısı rakam veri seti üzerinde OpenCV tabanlı görüntü ön işleme teknikleri uygulanarak elde edilen özellikler ile Yapay Sinir Ağı (Artificial Neural Network - ANN) eğitilmesini amaçlamaktadır. | ![Dlib](https://img.shields.io/badge/-Dlib-008000?logo=python&logoColor=white) ![FaceRec](https://img.shields.io/badge/-Face_Rec-blue) | [📂 Git](./Klasor_Adi) |
| **8** | ** DenseNet121 ile Zatürre Tespiti** | Bu proje, Derin Öğrenme (Deep Learning) ve Transfer Learning yöntemlerini kullanarak akciğer röntgeni (X-Ray) görüntüleri üzerinden zatürre teşhisi koymayı amaçlar. | ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![GANs](https://img.shields.io/badge/-GANs-purple) | [📂 Git](./Klasor_Adi) |
| **9** | **AI Destekli Squat Sayacı** | Bu proje, Python, OpenCV ve MediaPipe kütüphanelerini kullanarak gerçek zamanlı görüntü işleme ile squat egzersizlerini takip eden, açıları hesaplayan ve tekrarları otomatik olarak sayan bir uygulamadır. | ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) ![CNN](https://img.shields.io/badge/-CNN-grey) | [📂 Git](./Klasor_Adi) |
| **10** | **MediaPipe ile Gerçek Zamanlı Duygu Analizi** | Bu proje, Python, OpenCV ve MediaPipe kütüphanelerini kullanarak web kamerası üzerinden gerçek zamanlı yüz takibi (Face Mesh) yapar ve basit geometrik hesaplamalarla temel duyguları (Mutlu, Şaşkın, Nötr) tespit eder. | ![Tesseract](https://img.shields.io/badge/-Tesseract_OCR-black) ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white) | [📂 Git](./Klasor_Adi) |
| **11** | **U-Net ile Uydu Görüntülerini Bölütleme** | Bu proje, U-Net derin öğrenme mimarisini kullanarak hava görüntüleri (aerial imagery) üzerinde anlamsal segmentasyon (semantic segmentation) işlemini gerçekleştirir | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) ![Scikit](https://img.shields.io/badge/-Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white) | [📂 Git](./Klasor_Adi) |
| **12** | **PyTorch ile Sinirsel Stil Aktarımı** | Bir görüntünün (içerik) ana hatlarını korurken, başka bir görüntünün (stil) sanatsal dokusunu ve renklerini ona aktararak yeni bir sanat eseri oluşturur. | ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c) | [📂 Git](./Klasor_Adi) |
| **13** | **YOLO ile Trafik Levhaları Tespiti** | YOLOv8 kullanarak trafik levhalarını (hız sınırları, dur, girilmez vb.) gerçek zamanlı veya statik görüntüler üzerinde tespit etmek için geliştirilmiştir. | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) ![AI](https://img.shields.io/badge/-AI-red) | [📂 Git](./Klasor_Adi) |
| **14** | **YOLOv8 ile Araç Tespit ve Takip Sistemi** | Bu proje, Ultralytics YOLOv8 ve OpenCV kütüphanelerini kullanarak video üzerindeki araçları (veya diğer nesneleri) tespit eder ve takip eder  | ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white) ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | [📂 Git](./Klasor_Adi) |

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
