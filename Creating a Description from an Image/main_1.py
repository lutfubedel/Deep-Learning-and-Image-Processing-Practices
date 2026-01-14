# ==========================================
# KURULUM VE HAZIRLIK KOMUTLARI (Terminal)
# ==========================================
# Bu betiği çalıştırmadan önce terminalde şu komutları uyguladığınızdan emin olun:

# 1. Sanal ortam oluşturma (Opsiyonel ama önerilir):
# python -m venv venv
# .\venv\Scripts\activate

# 2. Gerekli kütüphanelerin yüklenmesi:
# pip install transformers torch pillow requests
# pip freeze > requirements.txt

# ==========================================
# KOD BAŞLANGICI
# ==========================================

# Gerekli kütüphaneleri içe aktarıyoruz
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch

# ---------------------------------------------------------
# 1. MODEL VE İŞLEMCİNİN (PROCESSOR) YÜKLENMESİ
# ---------------------------------------------------------
# Hugging Face Hub üzerinden Salesforce'un eğittiği BLIP modelini çekiyoruz.
# 'Processor': Görüntüyü modelin anlayabileceği matematiksel formata çevirir.
# 'Model': Görüntüden başlık (caption) üretme işini yapan yapay zeka modelidir.
print("Model yükleniyor, lütfen bekleyiniz...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# ---------------------------------------------------------
# 2. GÖRÜNTÜNÜN ALINMASI VE HAZIRLANMASI
# ---------------------------------------------------------
# Analiz edilecek resmin URL adresi
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"

# Resmi internetten indiriyoruz (stream=True) ve PIL kütüphanesi ile açıyoruz.
# .convert('RGB'): Resmin renk formatını garanti altına alıyoruz (Siyah-beyaz veya PNG şeffaflığı sorunu olmasın diye).
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# ---------------------------------------------------------
# 3. VERİ ÖN İŞLEME (PREPROCESSING)
# ---------------------------------------------------------
# Resmi işlemciden geçirerek PyTorch tensörlerine ("pt") dönüştürüyoruz.
# Bu aşamada resim, piksellerden oluşan sayısal matrislere dönüşür.
inputs = processor(raw_image, return_tensors="pt")

# ---------------------------------------------------------
# 4. MODEL TAHMİNİ (INFERENCE)
# ---------------------------------------------------------
# torch.no_grad(): Şu an eğitim (training) yapmadığımız, sadece tahmin (inference) 
# yaptığımız için gradyan hesaplamayı kapatıyoruz. Bu, bellek kullanımını azaltır ve hızı artırır.
with torch.no_grad():
    # Modeli girdilerle besliyoruz ve çıktı üretmesini istiyoruz.
    # out değişkeni, kelimelerin ID numaralarını (token IDs) tutar.
    out = model.generate(**inputs)

# ---------------------------------------------------------
# 5. ÇIKTIYI ÇÖZÜMLEME (DECODING)
# ---------------------------------------------------------
# Modelin ürettiği sayısal ID'leri tekrar insan okunabilir metne çeviriyoruz.
# skip_special_tokens=True: Cümle başı/sonu gibi teknik etiketleri temizler.
caption = processor.decode(out[0], skip_special_tokens=True)

# Sonucu ekrana yazdırıyoruz
print("-" * 30)
print(f"Resim Açıklaması: {caption}")
print("-" * 30)