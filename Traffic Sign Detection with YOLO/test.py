from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/traffic-sign-model/weights/best.pt")

image_path = "test_1.jpg"
image = cv2.imread(image_path)

results = model(image_path)[0]
print(results)

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    confidence = float(box.conf[0])
    label = f"{model.names[cls_id]} {confidence:.2f}"

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("test_1_detections.jpg", image)