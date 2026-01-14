# 🖼️ Image Captioning AI (Resimden Metin Üretme)

Bu proje, **Deep Learning** ve **Hugging Face Transformers** kütüphanesini kullanarak görselleri analiz eder ve içerikleri hakkında İngilizce açıklamalar (caption) üretir. Proje kapsamında iki farklı son teknoloji (State-of-the-Art) mimari kullanılmıştır.

## 🚀 Özellikler

* **İki Farklı Model Mimarisi:**
    1.  **BLIP (Bootstrapping Language-Image Pre-training):** Salesforce tarafından geliştirilen, görsel ve dilsel anlayışı birleştiren güçlü bir model.
    2.  **ViT-GPT2 (Vision Transformer + GPT-2):** Görüntü işleme için Vision Transformer (ViT) ve metin üretimi için GPT-2 kullanan hibrit yapı.
* **URL Desteği:** İnternet üzerindeki herhangi bir resim URL'si ile çalışabilir.
* **GPU Hızlandırma:** CUDA destekli donanımlarda (ViT-GPT2 scripti için) otomatik GPU kullanımı entegre edilmiştir.

## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

### 1. Depoyu Klonlayın veya İndirin
Proje dosyalarını bilgisayarınıza indirin ve proje dizinine gidin.

### 2. Sanal Ortam Oluşturun (Önerilen)
Sistem kütüphanelerinizi etkilememek için bir `venv` oluşturun:

```bash
# Windows için
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux için
python3 -m venv venv
source venv/bin/activate
