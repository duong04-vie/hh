from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

# ================== LOAD MODEL & CLASS LABELS ==================
yolo_model = YOLO('/Users/DELL/OneDrive/Desktop/noron/ALPR/ALPR/runs/detect/train2/weights/best.pt')  # Đường dẫn YOLO của bạn
cnn_model = load_model('viet_lp_char_cnn.h5')

# Load thứ tự lớp từ file JSON (không cần train_generator nữa)
with open('class_labels.json', 'r') as f:
    idx_to_class = json.load(f)

class_labels = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
print("Thứ tự lớp đã load:", class_labels)

# ================== XỬ LÝ ẢNH ==================
image_path = 'kq.jpg'
img = cv2.imread(image_path)
results = yolo_model.predict(img, conf=0.8, verbose=False)

for r in results:
    for box in r.boxes:
        if int(box.cls[0]) == 0:  # class 'bien_so'
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate = img[y1:y2, x1:x2]

            # Tiền xử lý
            plate = cv2.resize(plate, None, fx=2, fy=2)
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)

            # Tách ký tự
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            chars = []
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                if 10 < w < 200 and 20 < h < 200 and h/w < 5:
                    chars.append((x, y, w, h))

            chars = sorted(chars, key=lambda c: (c[1]//40, c[0]))  # Chia dòng + trái → phải

            # Dự đoán bằng CNN
            plate_text = ""
            for x,y,w,h in chars:
                char_img = gray[y:y+h, x:x+w]
                char_img = cv2.resize(char_img, (32,32))
                char_img = char_img.astype('float32') / 255.0
                char_img = np.expand_dims(char_img, axis=[0,-1])

                pred = cnn_model.predict(char_img, verbose=0)
                char = class_labels[np.argmax(pred)]
                plate_text += char

            print("Biển số:", plate_text)

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.putText(img, plate_text, (x1, y1-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 4)

cv2.imwrite('result_final.jpg', img)
cv2.imshow('Ket qua', img)
cv2.waitKey(0)
cv2.destroyAllWindows()