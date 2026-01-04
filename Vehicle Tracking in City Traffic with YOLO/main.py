# ==========================================
# KURULUM VE HAZIRLIK KOMUTLARI (Terminal)
# ==========================================
# python -m venv venv
# .\venv\Scripts\activate
# pip install ultralytics opencv-python
# pip freeze > requirements.txt
# Kaynak Veri Seti: https://www.kaggle.com/datasets/benjaminguerrieri/car-detection-videos

from ultralytics import YOLO
import cv2

# ==========================================
# 1. MODELİN VE VİDEONUN YÜKLENMESİ
# ==========================================

# YOLOv8 "Nano" modelini yüklüyoruz. 
# 'n' (nano): En hızlı ama en düşük hassasiyete sahip modeldir.
# Daha yüksek başarı için 'yolov8s.pt' (small) veya 'yolov8m.pt' (medium) denenebilir.
model = YOLO("yolov8n.pt") 

# İşlenecek videonun yolunu belirtiyoruz.
video_path = "videos/IMG_5268.MOV"
cap = cv2.VideoCapture(video_path)

# ==========================================
# 2. VİDEO KAYIT AYARLARI (Output)
# ==========================================

# Orijinal videonun genişlik, yükseklik ve FPS (saniyedeki kare sayısı) değerlerini alıyoruz.
# Böylece çıktı videosu orijinaliyle aynı özelliklerde olacak.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Videoyu kaydetmek için ayarlar:
# "output_video.avi": Kaydedilecek dosya adı.
# "XVID": Video sıkıştırma formatı (codec). Windows için genelde XVID veya MP4V uygundur.
output = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

# ==========================================
# 3. VİDEO İŞLEME DÖNGÜSÜ
# ==========================================

print("Video işleniyor... Çıkmak için 'q' tuşuna basınız.")

while cap.isOpened():
    # Videodan bir kare (frame) oku
    success, frame = cap.read()
    
    # Eğer kare okunamazsa (video bittiyse veya hata varsa) döngüyü kır
    if not success:
        break

    # --------------------------------------
    # Nesne Takibi (Tracking) İşlemi
    # --------------------------------------
    results = model.track(
        frame,
        persist=True,          # ÖNEMLİ: Nesne ID'lerinin (kimliklerinin) sonraki karelerde korunmasını sağlar.
        conf=0.3,              # Güven Eşiği: %30'un altındaki tahminleri yok say.
        iou=0.5,               # Çakışma Eşiği (Intersection over Union): Üst üste binen kutuların yönetimi.
        tracker="bytetrack.yaml" # Takip Algoritması: 'bytetrack' veya 'botsort' kullanılabilir.
    )

    # Tespit edilen nesneleri karenin üzerine çiz (Kutu, etiket ve olasılık değeri)
    # results[0] dememizin sebebi, modelin toplu işlem (batch) yapabilmesidir. 
    # Tek kare yolladığımız için ilk sonucu alıyoruz.
    annotated_frame = results[0].plot()

    # --------------------------------------
    # Görüntüleme ve Kayıt
    # --------------------------------------

    # İşlenmiş kareyi ekranda göster
    cv2.imshow("YOLOv8 Nesne Takibi", annotated_frame)
    
    # İşlenmiş kareyi video dosyasına yaz
    output.write(annotated_frame)

    # 'q' tuşuna basılırsa döngüden çık
    # waitKey(1): Her kare arasında 1 milisaniye bekle (Canlı akış hissi için)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# 4. TEMİZLİK (Kaynakları Serbest Bırakma)
# ==========================================
cap.release()        # Video okuma kaynağını serbest bırak
output.release()     # Kayıt dosyasını kapat ve kaydet
cv2.destroyAllWindows() # Açılan pencereleri kapat
print("İşlem tamamlandı.")