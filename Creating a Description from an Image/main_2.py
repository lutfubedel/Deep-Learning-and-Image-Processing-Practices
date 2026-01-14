# ==========================================
# GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ
# ==========================================

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import requests
import torch

# ---------------------------------------------------------
# 1. MODEL, İŞLEMCİ VE TOKENIZER YÜKLENMESİ
# ---------------------------------------------------------
# Bu model iki parçadan oluşur: Görüntüyü kodlayan (Encoder - ViT) ve metni çözen (Decoder - GPT2).
print("Model ve Tokenizer yükleniyor...")

# Ana modeli yüklüyoruz (Resimden Metne)
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Görüntü İşlemcisi: Resmi modelin anlayacağı matematiksel formata (tensor) çevirir.
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Tokenizer: Sayısal çıktıları (token ID) kelimelere (text) çevirir.
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# ---------------------------------------------------------
# 2. DONANIM AYARLAMASI (GPU vs CPU)
# ---------------------------------------------------------
# Bilgisayarda NVIDIA ekran kartı (CUDA) varsa onu kullanır, yoksa işlemciyi (CPU) kullanır.
# GPU kullanmak işlemi çok daha hızlandırır.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Kullanılan cihaz: {device}")

# Modeli seçilen donanıma taşıyoruz.
model.to(device)

# ---------------------------------------------------------
# 3. GÖRÜNTÜNÜN ALINMASI
# ---------------------------------------------------------
# URL'deki Kapadokya balon resmini çekiyoruz.
img_url = "https://media.istockphoto.com/id/1297349747/tr/foto%C4%9Fraf/t%C3%BCrkiyede-botan-kanyonu-%C3%BCzerinde-u%C3%A7an-s%C4%B1cak-hava-balonlar%C4%B1.jpg?s=1024x1024&w=is&k=20&c=ZSK39pZD9uUnUgvaIIvXmRVv_6MqQGIg0jVpM_o7Hag="

# Resmi açıp RGB formatına çeviriyoruz.
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# ---------------------------------------------------------
# 4. VERİ ÖN İŞLEME (PREPROCESSING)
# ---------------------------------------------------------
# Resmi işlemciden geçirip PyTorch tensörüne (pixel_values) çeviriyoruz.
# return_tensors="pt" -> PyTorch formatında döndür demektir.
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Resim verisini de modele gönderdiğimiz aynı donanıma (GPU veya CPU) gönderiyoruz.
pixel_values = pixel_values.to(device)

# ---------------------------------------------------------
# 5. MODELİN CÜMLE ÜRETMESİ (GENERATION)
# ---------------------------------------------------------
# Model resim verisini alıp bir açıklama üretiyor.
# max_length=16: Çok uzun saçmalamaması için maksimum 16 kelime (token) sınırı koyuyoruz.
# num_beams=4: (Opsiyonel ama önerilir) En iyi cümleyi bulmak için 4 farklı olasılığı aynı anda değerlendirir (Beam Search).
output_ids = model.generate(pixel_values, max_length=16, num_beams=4)

# ---------------------------------------------------------
# 6. ÇIKTIYI METNE ÇEVİRME (DECODING)
# ---------------------------------------------------------
# Üretilen sayısal ID'leri İngilizce metne çeviriyoruz.
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

print("-" * 30)
print(f"Resim Açıklaması: {caption}")
print("-" * 30)