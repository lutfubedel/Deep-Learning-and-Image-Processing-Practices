# ==========================================
# KURULUM VE HAZIRLIK KOMUTLARI (Terminal)
# ==========================================
# python -m venv venv
# .\venv\Scripts\activate
# pip install ultralytics opencv-python numpy
# # Kaynak Veri Seti: https://www.kaggle.com/datasets/benjaminguerrieri/car-detection-videos
# pip freeze > requirements.txt

import cv2
import numpy as np
from ultralytics import YOLO

def get_line_side(x, y, line_start, line_end):
    """
    Bir noktanın (x, y), belirtilen doğrunun (line_start -> line_end)
    sağında mı yoksa solunda mı (veya altında/üstünde) olduğunu belirler.
    
    Mantık: Vektörel çarpım (Cross Product) prensibine dayanır.
    Dönüş Değeri:
      Pozitif (+): Bir tarafta
      Negatif (-): Diğer tarafta
      Sıfır: Tam çizgi üzerinde
    """
    return np.sign((line_end[0] - line_start[0]) * (y - line_start[1]) - 
                   (line_end[1] - line_start[1]) * (x - line_start[0]))

# 1. MODEL VE VİDEO YÜKLEME
# ------------------------------------------
# YOLOv8 nano modelini yükle (en hızlı ama en düşük isabet oranı olan model)
model = YOLO("yolov8n.pt") 

# İşlenecek video dosyasını aç
cap = cv2.VideoCapture("videos/car/IMG_5268.MOV")

# Videonun başarıyla açılıp açılmadığını kontrol et
success, frame = cap.read()
if not success:
    exit("Video açılamadı veya dosya yolu hatalı.")

# 2. AYARLAR VE İLK DEĞERLER
# ------------------------------------------
# İşlem hızını artırmak için görüntüyü küçült (Orijinalin %60'ı)
frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
frame_height, frame_width = frame.shape[:2]

# Sanal sayım çizgisinin koordinatlarını belirle (Start: x,y - End: x,y)
# Bu koordinatlar videoya ve kamera açısına göre manuel ayarlanmalıdır.
line_start = (int(frame_width * 0.2), int(frame_height * 0.8)) # Örnek: Sol alt
line_end = (int(frame_width * 0.8), int(frame_height * 0.8))   # Örnek: Sağ alt

# Sadece saymak istediğimiz sınıfları ve sayaçları tanımla
counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0}

# Tekrar saymayı önlemek için sayılan araçların ID'lerini tutan küme
counted_ids = set()

# Araçların bir önceki karede çizginin hangi tarafında olduğunu tutan sözlük
# Format: {track_id: taraf_yönü (+1 veya -1)}
object_last_side = {}

# 3. ANA DÖNGÜ (HER KARE İÇİN)
# ------------------------------------------
while True:
    success, frame = cap.read()
    if not success:
        print("Video bitti veya okunamadı.")
        break # Video bittiyse döngüden çık
        
    # Görüntüyü baştaki ayarlarla aynı oranda küçült
    frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

    # YOLO TAKİP (TRACKING) İŞLEMİ
    # persist=True: Nesne ID'lerinin kareler arasında korunmasını sağlar.
    # tracker="bytetrack.yaml": Daha akıcı takip için ByteTrack algoritması kullanılır.
    results = model.track(
        frame,
        persist=True,
        stream=False,
        conf=0.5, # %50'nin altındaki tahminleri yoksay
        iou=0.5,
        tracker="bytetrack.yaml",
        verbose=False # Terminal çıktısını temiz tutmak için
    )

    # Eğer ekranda tespit edilen bir nesne varsa işlemlere başla
    if results[0].boxes.id is not None:
        # Tensör formatındaki verileri Python listelerine çevir
        ids = results[0].boxes.id.int().tolist()       # Takip ID'leri (1, 2, 3...)
        classes = results[0].boxes.cls.int().tolist()  # Sınıf ID'leri (0, 1, 2...)
        xyxy = results[0].boxes.xyxy                   # Koordinatlar (x1, y1, x2, y2)

        for i, box in enumerate(xyxy):
            cls_id = classes[i]
            track_id = ids[i]
            class_name = model.names[cls_id]

            # Eğer tespit edilen nesne saymak istediğimiz listede yoksa (örn: insan) atla
            if class_name not in counts:
                continue

            # Koordinatları al ve merkez noktasını (centroid) hesapla
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # 4. SAYIM MANTIĞI (ÇİZGİ GEÇİŞ KONTROLÜ)
            # ------------------------------------------
            # Nesnenin şu an çizginin hangi tarafında olduğunu bul
            current_side = get_line_side(cx, cy, line_start, line_end)
            
            # Nesnenin bir önceki karedeki tarafını hafızadan getir
            previous_side = object_last_side.get(track_id, None)
            
            # Mevcut durumu hafızaya kaydet (bir sonraki kare için 'önceki' olacak)
            object_last_side[track_id] = current_side

            # Eğer önceki kayıt varsa VE taraf değişmişse (işaret değişmişse)
            if previous_side is not None and previous_side != current_side:
                # Daha önce sayılmadıysa
                if track_id not in counted_ids:
                    counted_ids.add(track_id) # Sayılanlara ekle
                    counts[class_name] += 1   # İlgili aracın sayacını artır
                    
                    # Görsel geri bildirim: Geçiş anında çizgiyi kısa süre beyaz yapabiliriz (opsiyonel)
                    cv2.line(frame, line_start, line_end, (255, 255, 255), 3)

            # 5. GÖRSELLEŞTİRME
            # ------------------------------------------
            # Kutuyu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Etiketi ve ID'yi yaz
            cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, y1 - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # Merkez noktasını işaretle
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    # Sayım çizgisini ekrana çiz (Kırmızı)
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

    # 6. SONUÇLARI EKRANA YAZDIR
    # ------------------------------------------
    y_offset = 30
    for cls, count in counts.items():
        text = f"{cls} : {count}"
        # Arka plan için siyah bir kutucuk (okunabilirliği artırır)
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        # Ön plan yazısı (Beyaz)
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    # Görüntüyü göster
    cv2.imshow("Trafik Sayim Ekrani", frame)
    
    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik işlemleri
cap.release()
cv2.destroyAllWindows()