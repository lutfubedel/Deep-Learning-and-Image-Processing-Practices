# ==========================================
# KURULUM VE HAZIRLIK KOMUTLARI (Terminal)
# ==========================================

# python -m venv venv
# .\venv\Scripts\activate
# pip install mediapipe==0.10.9 opencv-python matplotlib
# pip freeze > requirements.txt
# https://www.kaggle.com/datasets/pashupatigupta/human-keypoints-tracking-dataset

import cv2
import mediapipe as mp
import numpy as np

# ==========================================
# 1. FONKSİYON: AÇI HESAPLAMA
# ==========================================
def calculate_angle(a, b, c):
    """
    Üç nokta (a, b, c) verildiğinde b noktası (köşe) etrafındaki açıyı hesaplar.
    a = İlk nokta (örneğin Kalça)
    b = Orta nokta (örneğin Diz)
    c = Son nokta (örneğin Ayak Bileği)
    """
    a = np.array(a) # İlk nokta koordinatları [x, y]
    b = np.array(b) # Orta nokta
    c = np.array(c) # Son nokta

    # arctan2 fonksiyonu ile radyan cinsinden açıyı buluyoruz
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    
    # Radyanı dereceye çeviriyoruz
    angle = np.abs(radians * 180.0 / np.pi)

    # Kol veya bacak açısı asla 180'den büyük olmamalı (iç açı hesaplaması)
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# ==========================================
# 2. AYARLAR VE DEĞİŞKENLER
# ==========================================
# MediaPipe çizim ve poz tahmin araçlarını hazırlıyoruz
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video kaynağını seçiyoruz. 
# "squat_test1.avi" yerine 0 yazarsanız webcam açılır.
cap = cv2.VideoCapture("squat_test1.avi")

# Sayaç değişkenleri
counter = 0        # Toplam squat sayısını tutar
stage = "Hazir"    # Hareketin durumunu tutar (Örn: "Asagi" veya "Yukari")

# ==========================================
# 3. ANA DÖNGÜ (VIDEO İŞLEME)
# ==========================================
# Pose modelini başlatıyoruz (Güvenilirlik eşiklerini 0.5 olarak belirledik)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read() # Videodan bir kare (frame) oku

        # Video bittiyse veya okunamadıysa döngüden çık
        if not ret:
            break

        # --- GÖRÜNTÜ FORMATI DÖNÜŞÜMÜ ---
        # OpenCV görüntüleri BGR (Mavi-Yeşil-Kırmızı) formatında okur.
        # MediaPipe ise RGB formatında çalışır. Bu yüzden dönüşüm yapıyoruz.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Performans optimizasyonu: Görüntüyü işlemden geçirirken yazmaya kapalı hale getiriyoruz.
        image.flags.writeable = False
        
        # MediaPipe ile iskelet tespiti yapılıyor
        results = pose.process(image) 
        
        # Çizim yapabilmek için görüntüyü tekrar yazılabilir yapıp BGR'ye çeviriyoruz
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- İSKELET NOKTALARINI ALMA VE AÇI HESABI ---
        try:
            # Tüm vücut noktalarını (landmarks) al
            landmarks = results.pose_landmarks.landmark

            # Gerekli 3 noktanın (Sol taraf) koordinatlarını çekiyoruz
            # MediaPipe'da koordinatlar 0 ile 1 arasında normalize edilmiştir.
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Dizdeki açıyı hesapla
            knee_angle = calculate_angle(hip, knee, ankle)

            # --- SQUAT MANTIĞI (LOGIC) ---
            # 1. Eğer açı 90 dereceden küçükse, kişi çömelmiştir (Down)
            if knee_angle < 90:
                stage = "Asagi" # Down
            
            # 2. Eğer açı 160 dereceden büyükse VE önceki durum "Asagi" ise
            # Kişi ayağa kalkmıştır, bir tekrar tamamlanmıştır.
            if knee_angle > 160 and stage == 'Asagi':
                stage = "Yukari" # Up
                counter += 1
                print(f"Squat Tamamlandi: {counter}")
            
            # --- GÖRSELLEŞTİRME (EKRAN ARAYÜZÜ) ---
            
            # Sol üst köşeye turuncu bir kutu çiz
            cv2.rectangle(image, (0,0), (240,110), (245,117,16), -1)

            # TEKRAR SAYISI (REPS)
            cv2.putText(image, 'TEKRAR', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # DURUM (STAGE - Aşağı/Yukarı)
            cv2.putText(image, 'DURUM', (90,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(stage), (90,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            
            # ANLIK AÇI DEĞERİ
            cv2.putText(image, f"Aci: {int(knee_angle)}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        except Exception as e:
            # Görüntüde insan tespit edilemezse kodun çökmemesi için hata yakalama
            # print("Hata:", e) # İsterseniz hatayı görmek için açabilirsiniz
            pass

        # --- İSKELET ÇİZİMİ ---
        # Tespit edilen noktaları ve bağlantı çizgilerini görüntüye çiz
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Nokta rengi
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Çizgi rengi
            )

        # Sonucu ekranda göster
        cv2.imshow('Squat Sayaci', image)

        # 'q' tuşuna basılırsa döngüyü kır ve çık
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Kaynakları serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()