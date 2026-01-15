# ==========================================
# KURULUM VE HAZIRLIK KOMUTLARI (Terminal)
# ==========================================
# python -m venv venv
# .\venv\Scripts\activate
# pip install torch torchvision pillow matplotlib tqdm
# pip freeze > requirements.txt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Cihaz yapılandırması: GPU varsa onu kullanır (çok daha hızlıdır), yoksa CPU'ya geçer.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

def load_image(image_path, max_size=None, shape=None):
    """
    Görüntüyü diskten okur, yeniden boyutlandırır ve VGG modelinin
    anlayabileceği Tensor formatına dönüştürüp normalize eder.
    """
    image = Image.open(image_path).convert('RGB')

    # Görüntü boyutunu ayarlama mantığı
    if shape is not None:
        size = shape # Eğer belirli bir şekil (shape) verildiyse onu kullan
    else:
        # Maksimum boyutu aşmamak için kontrol
        size = max(image.size)
        if max_size is not None and size > max_size:
            size = max_size 
    
    # Görüntü Ön İşleme (Preprocessing) İşlemleri
    in_transform = transforms.Compose([
        transforms.Resize((size, size) if isinstance(size, int) else size), # Boyutlandırma
        transforms.ToTensor(), # Görüntüyü 0-255 arasından 0-1 arasına çeker ve Tensor yapar
        # Normalizasyon: VGG19'un eğitildiği ImageNet verisetinin ortalama ve standart sapma değerleri.
        # Bu işlem, modelin görüntüyü doğru tanıması için zorunludur.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Batch boyutunu ekle (1, 3, H, W) ve cihaza (GPU/CPU) gönder
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image.to(device)

def im_convert(tensor):
    """
    Modelin çıktısı olan Tensor'u alır, normalizasyonu tersine çevirir
    ve insanların görebileceği (Matplotlib ile çizilebilecek) bir resme dönüştürür.
    """
    image = tensor.clone().detach().cpu().squeeze(0) # Bellekten ayır ve batch boyutunu kaldır
    
    # Normalizasyonu geri alma (Un-normalize): x * std + mean
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    
    image = image.clamp(0, 1) # Değerlerin 0 ile 1 arasında kalmasını garanti et
    return image.permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C) formatına çevir

def gram_matrix(tensor):
    """
    Stil transferinin kalbi burasıdır.
    Bir özellik haritasının (feature map) kendisiyle olan korelasyonunu hesaplar.
    Bu matris, görüntünün 'içeriğini' değil, 'dokusunu/stilini' temsil eder.
    """
    _, d, h, w = tensor.size() # d=derinlik (kanal sayısı), h=yükseklik, w=genişlik
    tensor = tensor.view(d, h * w) # Vektör haline getir
    gram = torch.mm(tensor, tensor.t()) # Matris çarpımı (Korelasyon hesaplama)
    return gram

class VVGGFeatures(nn.Module):
    def __init__(self):
        super(VVGGFeatures, self).__init__()

        # Önceden eğitilmiş (Pretrained) VGG19 modelini indiriyoruz.
        # Sadece özellik çıkarma (features) kısmını alıyoruz, sınıflandırma (classifier) kısmını atıyoruz.
        self.vgg = models.vgg19(pretrained=True).features[:29].to(device).eval()
        
        # Modelin ağırlıklarını donduruyoruz. Çünkü modeli eğitmeyeceğiz,
        # sadece resmi güncelleyeceğiz.
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Hangi katmanlardan çıktı alacağımızı belirliyoruz.
        # Derinlere indikçe özellikler karmaşıklaşır.
        self.layers = {
            "0": "conv1_1",   # Stil
            "5": "conv2_1",   # Stil
            "10": "conv3_1",  # Stil
            "19": "conv4_1",  # Stil
            "21": "conv4_2",  # İÇERİK (Content) katmanı - Genellikle biraz derin seçilir
            "28": "conv5_1",  # Stil
        }

    def forward(self, x):
        features = {}
        # Görüntüyü katman katman ilerletiyoruz
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            # Eğer bu katman bizim istediğimiz katmanlardan biriyse, çıktısını sakla
            if name in self.layers:
                features[self.layers[name]] = x
        return features

def run_style_transfer(content_img, 
                       style_img, 
                       steps=2000, 
                       style_weight=1e6, # Stilin baskınlığını belirler (Genelde yüksek verilir)
                       content_weight=1): # İçeriğin korunma miktarını belirler
    
    # Başlangıçta hedef görüntü, içerik görüntüsünün bir kopyasıdır.
    # requires_grad=True: Çünkü biz bu görüntünün piksellerini değiştireceğiz/eğiteceğiz.
    target = content_img.clone().requires_grad_(True).to(device)
    
    # Optimizer olarak Adam kullanıyoruz, hedef görüntüyü güncelleyecek.
    optimizer = optim.Adam([target], lr=0.003)
    model = VVGGFeatures()

    print("Stil transferi başlatılıyor...")
    
    # İçerik ve Stil görüntülerinin özelliklerini bir kere hesapla (sabit kalacaklar)
    content_features = model(content_img)
    style_features = model(style_img)

    for step in tqdm(range(steps)):
        # Hedef görüntünün o anki özelliklerini çıkar
        target_features = model(target)

        # --- CONTENT LOSS (İçerik Kaybı) ---
        # Hedef görüntü ile Orijinal içerik görüntüsünün 'conv4_2' katmanındaki farkı
        content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"])**2)

        # --- STYLE LOSS (Stil Kaybı) ---
        style_loss = 0
        for layer in ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]:
            target_feature = target_features[layer]
            style_feature = style_features[layer]

            # Stil karşılaştırması Gram Matrisleri üzerinden yapılır
            target_gram = gram_matrix(target_feature)
            style_gram = gram_matrix(style_feature)

            layer_style_loss = torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss 

        # Toplam Kayıp (Total Loss)
        total_loss = content_weight * content_loss + style_weight * style_loss

        # Geri Yayılım (Backpropagation) ve Güncelleme
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Adım {step}, Toplam Kayıp: {total_loss.item():.2f}")
            
    return target

# ==========================================
# ÇALIŞTIRMA BÖLÜMÜ
# ==========================================
# Not: 'content.jpeg' ve 'style.jpg' dosyalarının proje klasöründe olduğundan emin olun.

try:
    # İçerik görüntüsünü yükle
    content = load_image("content.jpeg", max_size=400)
    
    # Stil görüntüsünü yükle (İçerik görüntüsüyle aynı boyutta olması işlem kolaylığı sağlar)
    style = load_image("style.jpg", shape=tuple(content.shape[-2:]))

    # Transfer işlemini başlat
    output = run_style_transfer(content, style, steps=2000)

    # Sonucu göster
    plt.figure(figsize=(10,5))
    plt.imshow(im_convert(output))
    plt.title("Stilize Edilmiş Görüntü")
    plt.axis('off')
    plt.show()

except FileNotFoundError:
    print("HATA: 'content.jpeg' veya 'style.jpg' bulunamadı. Lütfen resim yollarını kontrol edin.")