# ==========================================
# KURULUM VE HAZIRLIK KOMUTLARI (Terminal)
# ==========================================
# python -m venv venv
# .\venv\Scripts\activate
# pip install ultralytics
# pip freeze > requirements.txt
# https://universe.roboflow.com/university-km5u7/traffic-sign-detection-yolov8-awuus/dataset/11

from ultralytics import YOLO

# ==========================================
# 1. MODELİN YÜKLENMESİ
# ==========================================

# "yolov8n.pt": YOLOv8'in "Nano" versiyonudur. En hızlı ama en düşük isabet oranlı modeldir.
model = YOLO("yolov8n.pt") 

# ==========================================
# 2. EĞİTİM (TRAINING) PARAMETRELERİ
# ==========================================
model.train(
    # Veri seti konfigürasyon dosyası. Eğitim ve test resimlerinin yolları ile sınıf isimleri burada bulunur.
    data ="traffic-sign-detection/data.yaml",
    
    # Modelin tüm veri setini kaç kez göreceği. 
    # Not: 2 epoch sadece test içindir, gerçek bir eğitim için genelde 50-100 arası önerilir.
    epochs=2,
    
    # Eğitim sırasında resimlerin yeniden boyutlandırılacağı çözünürlük (piksel cinsinden, kare format).
    imgsz=640,
    
    # Modelin ağırlıklarını güncellemeden önce aynı anda işleyeceği resim sayısı.
    # GPU/CPU belleğiniz yetersiz gelirse bu sayıyı düşürün (örn: 8 veya 4).
    batch=16,
    
    # Proje ismi. Sonuçlar 'runs/detect/traffic-sign-model' klasörüne kaydedilir.
    name="traffic-sign-model",
    
    # Başlangıç öğrenme oranı (Learning Rate). Modelin ne kadar "hızlı" öğreneceğini belirler.
    lr0=0.01,
    
    # Optimizasyon algoritması. "SGD" (Stochastic Gradient Descent) veya "Adam" sık kullanılır.
    optimizer="SGD",
    
    # Ağırlık azaltma (Weight Decay). Modelin aşırı öğrenmesini (overfitting) engellemek için ceza katsayısı.
    weight_decay=0.0005,
    
    # Momentum. Optimizasyon sırasında yerel minimumlara takılmayı önlemeye yardımcı olur.
    momentum=0.935,
    
    # Erken Durdurma (Early Stopping). 
    # Eğer model 50 epoch boyunca iyileşme göstermezse eğitimi otomatik durdurur (Vakit kaybını önler).
    patience=50,
    
    # Veri yükleme (Data Loading) için kullanılacak işlemci çekirdeği sayısı.
    # Windows'ta bazen hata verebilir, hata alırsanız 0 yapın.
    workers=2,
    
    # Eğitimin yapılacağı donanım. 
    # "cpu": İşlemci kullanır (Çok yavaştır).
    # "0": NVIDIA GPU kullanır (Varsa mutlaka bunu kullanın).
    # "mps": Mac (M1/M2 çip) için hızlandırıcı.
    device="cpu",
    
    # Eğitim sonucunda model ağırlıklarını (.pt dosyaları) kaydet.
    save = True,
    
    # Checkpoint kaydetme sıklığı. Her 1 epoch'ta bir modelin yedeğini alır.
    save_period=1,
    
    # Her epoch sonunda doğrulama (validation) setinde test yap.
    val = True,
    
    # Terminal ekranında detaylı log/bilgi gösterimi yap.
    verbose = True,
)