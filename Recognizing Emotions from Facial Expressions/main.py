# ==========================================
# KURULUM VE HAZIRLIK KOMUTLARI (Terminal)
# ==========================================
# Gerekli kütüphaneleri yüklemek için terminalde çalıştırılacak komutlar:
# python -m venv venv
# .\venv\Scripts\activate
# pip install mediapipe==0.10.9 opencv-python matplotlib
# pip freeze > requirements.txt

import cv2
import mediapipe as mp
import numpy as np

# ==========================================
# 1. MEDIAPIPE AYARLARI
# ==========================================
# Yüz ağı (Face Mesh) çözümünü başlatıyoruz.
mp_face_mesh = mp.solutions.face_mesh

# FaceMesh modelini yapılandırıyoruz:
# static_image_mode=False: Video akışı olduğu için False (daha hızlı takip sağlar).
# max_num_faces=1: Sadece 1 yüz algıla (performans için).
# refine_landmarks=True: Göz bebekleri gibi daha detaylı noktaları da getirir.
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5, # %50 güvenin altındaysa yüzü tekrar aramaya başlar.
    min_tracking_confidence=0.5   # Takip güveni %50'nin altına düşerse algılamayı yeniler.
)

# Kamerayı başlat (0 varsayılan web kamerasını temsil eder)
cap = cv2.VideoCapture(0)

# ==========================================
# 2. YÜZ NOKTA İNDEXLERİ (LANDMARKS)
# ==========================================
# MediaPipe 468 tane nokta döndürür. Biz sadece ilgili noktaların indekslerini seçiyoruz.
# Bu numaralar MediaPipe'ın standart yüz haritasına göredir.
LEFT_EYE = [159, 145]       # Sol gözün üst ve alt kapak noktaları
MOUTH = [61, 291]           # Ağzın sol ve sağ köşe noktaları
LEFT_EYEBROW = [65, 158]    # Kaşın bir noktası ve gözün üstü (Kaş kaldırmayı ölçmek için)

def detect_emotion(landmarks, img_w, img_h):
    """
    Yüzdeki noktalara bakarak basit geometrik hesaplarla duygu tahmini yapar.
    """

    # Yardımcı Fonksiyon: Koordinat Dönüştürme
    # MediaPipe noktaları 0.0 ile 1.0 arasında verir (normalize edilmiş).
    # Bunları ekranın piksel boyutuna (örn: 1920x1080) çevirmemiz gerekir.
    def get_point(index):
        lm = landmarks[index]
        # x * genişlik, y * yükseklik işlemi ile gerçek piksel konumunu buluyoruz.
        return np.array([int(lm.x * img_w), int(lm.y * img_h)])
    
    # --- Şaşkınlık Hesabı (Kaş Kaldırma) ---
    brow_point = get_point(LEFT_EYEBROW[0]) # Kaş noktası
    eye_point = get_point(LEFT_EYE[0])      # Göz noktası
    
    # İki nokta arasındaki Öklid mesafesini (kuş uçuşu uzaklık) hesapla
    brow_lift = np.linalg.norm(brow_point - eye_point)

    # --- Mutluluk Hesabı (Gülümseme Genişliği) ---
    mouth_left = get_point(MOUTH[0])   # Ağız sol köşe
    mouth_right = get_point(MOUTH[1])  # Ağız sağ köşe
    
    # Ağız genişliğini hesapla
    mouth_width = np.linalg.norm(mouth_left - mouth_right)

    # --- Karar Mekanizması (Eşik Değerleri) ---
    # NOT: Bu değerler (25 ve 60) kameraya olan uzaklığa göre değişebilir.
    # Daha gelişmiş sistemlerde oransal hesaplama (örn: yüz genişliğine oranı) kullanılır.
    
    if brow_lift > 25: # Eğer kaş ile göz arası 25 pikselden fazlaysa
        return "Surprised" 
    elif mouth_width > 60: # Eğer ağız genişliği 60 pikselden fazlaysa
        return "Happy" 
    else:
        return "Neutral"

# ==========================================
# 3. ANA DÖNGÜ (VIDEO AKIŞI)
# ==========================================
while True:
    success, frame = cap.read()
    if not success:
        print("Kamera bulunamadı veya akış bitti.")
        break
    
    # MediaPipe RGB formatında çalışır, OpenCV BGR kullanır. Dönüşüm yapıyoruz.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Görüntüyü işleyip sonuçları alıyoruz
    results = face_mesh.process(rgb_frame)

    # Görüntü boyutlarını al (Yükseklik, Genişlik, Kanal sayısı)
    h, w, _ = frame.shape

    # Eğer ekranda en az bir yüz tespit edildiyse:
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Duygu tahmini fonksiyonunu çağır
            emotion = detect_emotion(face_landmarks.landmark, w, h)
            
            # Sonucu ekrana yazdır (Sol üst köşe)
            cv2.putText(frame, f'Emotion: {emotion}', (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Yüzdeki ağ yapısını (mesh) yeşil çizgilerle ekrana çiz
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_TESSELATION, # Yüz üçgenleme ağı
                landmark_drawing_spec=None,        # Noktaları çizme (sade görünüm için kapalı)
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1) # Bağlantı çizgileri yeşil
            )

    # İşlenmiş kareyi ekranda göster
    cv2.imshow('Emotion Detection', frame)
    
    # ESC tuşuna basılırsa döngüyü kır (27 = ESC)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Kaynakları serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()