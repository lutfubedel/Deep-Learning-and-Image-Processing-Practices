# ==========================================
# KURULUM VE HAZIRLIK KOMUTLARI (Terminal)
# ==========================================
# python -m venv venv
# .\venv\Scripts\activate
# pip install ultralytics opencv-python numpy
# # Veri Seti: https://www.kaggle.com/datasets/khitthanhnguynphan/crowduit?select=Crowd-UIT
# pip freeze > requirements.txt

import cv2
import numpy as np
from ultralytics import YOLO

# 1. MODEL VE VİDEO YÜKLEME
# ------------------------------------------
# YOLOv8 nano modelini yükle (İnsan tespiti için yeterli ve hızlı)
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("videos/people/2.mp4")

# Videonun açılıp açılmadığını kontrol et
success, frame = cap.read()
if not success: 
    exit("Video Çalışmıyor veya dosya yolu hatalı.")

# 2. AYARLAR VE İLK DEĞERLER
# ------------------------------------------
# İlk kareyi okuyup boyutları alıyoruz (Referans çizgisi için gerekli)
frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
frame_height, frame_width = frame.shape[:2]

# Dikey referans çizgisini ekranın tam ortasına konumlandır
# Bu çizgi, "Giren" ve "Çıkan" ayrımını yapacağımız sınır kapısıdır.
line_x = int(frame_width * 0.5)

# Sayaç değişkenleri
giren = 0  # Soldan sağa geçenler
cikan = 0  # Sağdan sola geçenler

# Mükerrer sayımı önlemek için ID takibi
counted_ids = set()

# Kişilerin bir önceki karedeki X koordinatlarını saklayan sözlük
# {track_id: önceki_x_konumu}
person_last_x = {}

# 3. ANA DÖNGÜ
# ------------------------------------------
while True:
    success, frame = cap.read()
    if not success:
        print("Video bitti.")
        break
        
    # Görüntüyü küçült (İşlem hızını artırır)
    frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)

    # YOLO TAKİP (TRACKING)
    # persist=True: Nesne kaybolsa bile ID'sini hatırlamaya çalışır
    results = model.track(
        frame,
        persist=True,
        stream=False,
        conf=0.25, # %25 güven eşiği
        iou=0.3,
        tracker="bytetrack.yaml",
        verbose=False # Konsol kirliliğini önle
    )

    # Tespit varsa işle
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().tolist()
        classes = results[0].boxes.cls.int().tolist()
        xyxy = results[0].boxes.xyxy

        for i, box in enumerate(xyxy):
            cls_id = classes[i]
            track_id = ids[i]
            class_name = model.names[cls_id]

            # Sadece "person" (insan) sınıfını işle, diğerlerini atla
            if class_name != "person":
                continue

            # Koordinatları al ve merkez noktayı (centroid) bul
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2) # Nesnenin yatay merkezi
            cy = int((y1 + y2) / 2) # Nesnenin dikey merkezi

            # 4. SAYIM MANTIĞI (SAĞA/SOLA GEÇİŞ)
            # ------------------------------------------
            # Kişinin önceki X konumunu hafızadan getir
            previous_x = person_last_x.get(track_id, None)
            
            # Şimdiki X konumunu hafızaya kaydet (Bir sonraki kare için)
            person_last_x[track_id] = cx

            # Eğer geçmiş verisi varsa karşılaştırma yap
            if previous_x is not None:
                
                # DURUM 1: SAĞDAN SOLA GEÇİŞ (ÇIKAN)
                # Önceki konumu çizginin sağında (> line_x) VE şu anki konumu solunda veya üstünde (<= line_x)
                if previous_x > line_x >= cx:
                    if track_id not in counted_ids:
                        cikan += 1
                        counted_ids.add(track_id)
                        # Görsel efekt: Geçiş anında çizgiyi kırmızı yap
                        cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 0, 255), 4)

                # DURUM 2: SOLDAN SAĞA GEÇİŞ (GİREN)
                # Önceki konumu çizginin solunda (< line_x) VE şu anki konumu sağında veya üstünde (>= line_x)
                elif previous_x < line_x <= cx:
                     if track_id not in counted_ids:
                        giren += 1
                        counted_ids.add(track_id)
                        # Görsel efekt: Geçiş anında çizgiyi yeşil yap
                        cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 255, 0), 4)
            
            # Görselleştirme (Kutu, ID ve Merkez Nokta)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID : {track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
    
    # Referans çizgisini çiz (Varsayılan Kırmızı)
    cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 0, 255), 2)

    # Sayaçları ekrana yazdır
    cv2.putText(frame, f"Giren (saga) : {giren}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Cikan (sola) : {cikan}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()