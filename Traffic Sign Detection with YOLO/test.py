from ultralytics import YOLO
import cv2

# ==========================================
# 1. EĞİTİLMİŞ MODELİ YÜKLEME
# ==========================================
# Eğitim bittikten sonra "runs/detect/..." klasöründe oluşan ağırlıkları kullanıyoruz.
# "best.pt": Eğitim boyunca en yüksek başarıyı (mAP) gösteren ağırlık dosyasıdır.
# "last.pt": Eğitimin en son epoch'undaki ağırlıklardır (Genelde best.pt tercih edilir).
model = YOLO("runs/detect/traffic-sign-model/weights/best.pt")

# ==========================================
# 2. TEST GÖRÜNTÜSÜNÜ HAZIRLAMA
# ==========================================
image_path = "test_1.jpg"

# Görüntüyü OpenCV ile okuyoruz (Çizim yapmak için gerekli).
# OpenCV görüntüleri BGR (Mavi-Yeşil-Kırmızı) formatında okur.
image = cv2.imread(image_path)

# ==========================================
# 3. TAHMİN (INFERENCE) YAPMA
# ==========================================
# Modeli resim üzerinde çalıştırıyoruz.
# [0] dememizin sebebi, modelin bir liste döndürmesidir (batch işlemleri için).
# Biz tek resim verdiğimiz için listenin ilk elemanını alıyoruz.
results = model(image_path)[0]

# Sonuçların ham verisini terminale yazdırır (Tensor bilgileri, hız vb.).
print(results)

# ==========================================
# 4. SONUÇLARI GÖRSELLEŞTİRME
# ==========================================
# Tespit edilen her bir nesne (kutu) için döngü başlatıyoruz.
for box in results.boxes:
    # --- Koordinatları Alma ---
    # xyxy: x1 (sol), y1 (üst), x2 (sağ), y2 (alt) koordinatlarıdır.
    # map(int, ...): Koordinatlar float (ondalık) gelir, çizim için int (tam sayı) yaparız.
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # --- Sınıf ve Güven Skoru ---
    # cls: Tespit edilen nesnenin sınıf ID'si (0, 1, 2 gibi).
    cls_id = int(box.cls[0])
    # conf: Modelin tahmininden ne kadar emin olduğu (0.0 ile 1.0 arası).
    confidence = float(box.conf[0])
    
    # Ekrana yazılacak metni oluşturuyoruz (Örn: "Dur Levhası 0.85").
    # model.names[cls_id]: ID'nin karşılık geldiği etiket ismini (string) getirir.
    label = f"{model.names[cls_id]} {confidence:.2f}"

    # --- Çizim İşlemleri (OpenCV) ---
    # Dikdörtgen çizme: (Resim, Başlangıç, Bitiş, Renk(B,G,R), Kalınlık)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Metin yazma: (Resim, Metin, Konum, Font, Boyut, Renk, Kalınlık)
    # y1 - 10: Yazıyı kutunun biraz üzerine yazar.
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# ==========================================
# 5. GÖSTERME VE KAYDETME
# ==========================================

# Resmi "Detections" adında bir pencerede açar.
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sonuç çizilmiş resmi bilgisayara kaydeder.
cv2.imwrite("test_1_detections.jpg", image)