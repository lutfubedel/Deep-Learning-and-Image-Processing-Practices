# 🖼️ Image Captioning AI (Resimden Metin Üretme)

Bu proje, **Deep Learning** ve **Hugging Face Transformers** kütüphanesini kullanarak görselleri analiz eder ve içerikleri hakkında İngilizce açıklamalar (caption) üretir. Proje kapsamında iki farklı son teknoloji (State-of-the-Art) mimari kullanılmıştır.

## 🚀 Özellikler

* **İki Farklı Model Mimarisi:**
    1.  **BLIP (Bootstrapping Language-Image Pre-training):** Salesforce tarafından geliştirilen, görsel ve dilsel anlayışı birleştiren güçlü bir model.
    2.  **ViT-GPT2 (Vision Transformer + GPT-2):** Görüntü işleme için Vision Transformer (ViT) ve metin üretimi için GPT-2 kullanan hibrit yapı.
* **URL Desteği:** İnternet üzerindeki herhangi bir resim URL'si ile çalışabilir.
* **GPU Hızlandırma:** CUDA destekli donanımlarda (ViT-GPT2 scripti için) otomatik GPU kullanımı entegre edilmiştir.

## 📸 Sonuçlar

Modelin örnek çıktıları aşağıda verilmiştir

<table>
  <tr>
    <td width="50%"><img src="https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg" width="100%"></td>
    <td width="50%"><img src="images/imag_2.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>A woman sitting on the beach with her dog</b></td>
    <td align="center"><b>Tespit Sonucu</b></td>
  </tr>
</table>

