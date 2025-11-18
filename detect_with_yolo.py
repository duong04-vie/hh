import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from skimage import measure
from imutils import perspective
import imutils

# --- 1. CNN Model cho Recognition (train trước với dataset ký tự) ---
def build_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='softmax'))  # 31 ký tự + background

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Ánh xạ lớp sang ký tự
ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L',
              10: 'M', 11: 'N', 12: 'P', 13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X',
              19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3', 25: '4', 26: '5',
              27: '6', 28: '7', 29: '8', 30: '9', 31: 'BG'}

# Tải model CNN đã train (bạn cần train trước)
cnn_model = build_cnn_model()
cnn_model.load_weights('cnn_license_plate_weights.h5')  # Thay bằng file weights của bạn

# --- 2. Cấu hình YOLO ---
MODEL_PATH = '/Users/DELL/OneDrive/Desktop/noron/ALPR/ALPR/runs/detect/train2/weights/best.pt'
LICENSE_PLATE_CLASS_NAME = 'bien_so'
CONFIDENCE_THRESHOLD = 0.8
IMAGE_PATH = 'kq.jpg'

model = YOLO(MODEL_PATH)
LICENSE_PLATE_CLASS_ID = list(model.names.keys())[list(model.names.values()).index(LICENSE_PLATE_CLASS_NAME)]

# Tải ảnh
image = cv2.imread(IMAGE_PATH)

# --- 3. Phát hiện biển số bằng YOLO ---
results = model.predict(source=image, conf=CONFIDENCE_THRESHOLD, verbose=False)
found_plates = []

for r in results:
    boxes = r.boxes
    for box in boxes:
        if int(box.cls[0]) == LICENSE_PLATE_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(float(box.conf[0]), 2)

            # Crop vùng biển số
            plate_crop = image[y1:y2, x1:x2]

            # --- 4. Tiền xử lý và Bird's Eye View (giảm méo) ---
            # Giả sử lấy 4 điểm (pts) từ detection hoặc hardcode cho ví dụ
            pts = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype='float32')  # Thay bằng tọa độ chính xác nếu có
            plate_crop = perspective.four_point_transform(plate_crop, pts)

            # Chuyển HSV và threshold
            V = cv2.split(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV))[2]
            T = measure.filters.threshold_local(V, 15, offset=10, method="gaussian")
            thresh = (V > T).astype("uint8") * 255
            thresh = cv2.bitwise_not(thresh)
            thresh = imutils.resize(thresh, width=400)
            thresh = cv2.medianBlur(thresh, 5)

            # --- 5. Segmentation bằng CCA ---
            labels = measure.label(thresh, connectivity=2, background=0)
            char_candidates = []

            for label in np.unique(labels):
                if label == 0: continue
                labelMask = np.zeros(thresh.shape, dtype="uint8")
                labelMask[labels == label] = 255
                cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if len(cnts) > 0:
                    c = max(cnts, key=cv2.contourArea)
                    (cx, cy, cw, ch) = cv2.boundingRect(c)
                    aspectRatio = cw / float(ch)
                    solidity = cv2.contourArea(c) / float(cw * ch)
                    heightRatio = ch / float(plate_crop.shape[0])

                    if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:
                        candidate = thresh[cy:cy + ch, cx:cx + cw]
                        square_candidate = cv2.resize(candidate, (28, 28), cv2.INTER_AREA)
                        square_candidate = square_candidate.reshape((28, 28, 1)) / 255.0
                        char_candidates.append((square_candidate, (cy, cx)))

            # --- 6. Recognition bằng CNN và Format ---
            plate_text = ""
            if char_candidates:
                # Dự đoán từng ký tự
                predictions = cnn_model.predict(np.array([c[0] for c in char_candidates]))
                chars = [ALPHA_DICT[np.argmax(p)] for p in predictions if ALPHA_DICT[np.argmax(p)] != 'BG']

                # Sắp xếp theo y (dòng) rồi x
                char_candidates.sort(key=lambda x: (x[1][0], x[1][1]))  # y trước, x sau
                first_line, second_line = [], []
                y_threshold = char_candidates[0][1][0] + 40  # Ngưỡng phân dòng
                for i, (cand, (y, x)) in enumerate(char_candidates):
                    char = chars[i]
                    if y < y_threshold:
                        first_line.append(char)
                    else:
                        second_line.append(char)

                plate_text = "".join(first_line) + "".join(second_line)  # Hoặc thêm '-' nếu cần

            found_plates.append((plate_text, confidence, (x1, y1, x2, y2)))

# --- 7. Vẽ kết quả ---
for text, conf, (x1, y1, x2, y2) in found_plates:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(image, f"{text} ({conf})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imwrite('ket_qua_bien_so_cnn.jpg', image)
print(f"Kết quả: {plate_text}")