import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Eğitilmiş modeli yükle
print("Model yükleniyor...")
model = load_model('mnist_cnn_model.h5')
print("Model yüklendi!")

# 2. Webcam'i başlat 
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare (frame) oku
    success, frame = cap.read()
    if not success:
        break

    # Görüntüyü gri tona çevir (Modelimiz renkli görmüyor)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Ekran boyutlarını al
    h, w = gray.shape
    
    # Odaklanılacak kutunun boyutu (Buraya rakamı denk getirmelisiniz)
    box_size = 200

    # Kutunun koordinatlarını hesapla (Ekranın tam ortası)
    top_left = (w//2 - box_size//2, h//2 - box_size//2)
    bottom_right = (w//2 + box_size//2, h//2 + box_size//2)
    
    # Ekrana yeşil bir çerçeve çiz (Kullanıcı nereye yazacağını bilsin)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # 3. İLGİ ALANI (ROI - Region of Interest) KESME VE İŞLEME
    # Sadece kutunun içindeki görüntüyü al
    roi = gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    # -- İYİLEŞTİRME: Eşikleme (Thresholding) --
    # Kameradaki gölgeleri temizlemek ve yazıyı netleştirmek için siyah-beyaz ayrımı yap.
    # 100'den koyu olan yerleri tam siyah (0), açık olanları tam beyaz (255) yapar.
    # Not: Eğitim setinizde "Siyah Yazı / Beyaz Arka Plan" kullandığınız için THRESH_BINARY uygundur.
    _, roi_thresh = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)

    # Modelin beklediği boyuta getir (28x28 piksel)
    roi_resized = cv2.resize(roi_thresh, (28, 28))
    
    # Normalizasyon (0-255 arasını 0-1 arasına çek)
    roi_normalized = roi_resized.astype('float32') / 255.0
    
    # Modelin beklediği şekle sok: (1 adet, 28 genişlik, 28 yükseklik, 1 kanal)
    roi_input = roi_normalized.reshape((1, 28, 28, 1))

    # 4. TAHMİN (PREDICTION)
    # verbose=0: Konsola sürekli çıktı basmasını engeller, hız kazandırır.
    predictions = model.predict(roi_input, verbose=0)
    
    # En yüksek olasılığa sahip indeksi (rakamı) al
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions) # Güven oranını al

    # 5. SONUCU EKRANA YAZDIR
    display_text = f'Tahmin: {predicted_digit} (Guven: %{confidence*100:.0f})'
    cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Ana kamera görüntüsünü göster
    cv2.imshow("Kamera", frame)
    
    # Modelin ne gördüğünü anlamak için küçük pencere (Debugging için çok yararlıdır)
    cv2.imshow("Modelin Goruşu", roi_resized)

    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik işlemleri
cap.release()
cv2.destroyAllWindows()