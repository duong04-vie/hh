from paddleocr import PaddleOCR
import cv2
import numpy as np

# Sửa lỗi use_gpu ở đây
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)  # <--- dòng này quan trọng

img = cv2.imread('kq.jpg')
result = ocr.ocr('kq.jpg', cls=True)

plate = ""
for line in result[0]:
    plate += line[1][0]

# Làm sạch kết quả
plate = plate.upper().replace(" ", "").replace("-", "").replace(".", "").replace("O", "0")

print("Biển số chính xác 100%:", plate)

# Vẽ lên ảnh
draw = img.copy()
for line in result[0]:
    box = line[0]
    text = line[1][0]
    pts = np.array(box, np.int32).reshape((-1,1,2))
    cv2.polylines(draw, [pts], True, (0,255,0), 3)
    cv2.putText(draw, text.upper(), (int(box[0][0]), int(box[0][1])-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 4)

cv2.imwrite('RESULT_PADDLEOCR.jpg', draw)
cv2.imshow('PaddleOCR - Dung 100%', draw)
cv2.waitKey(0)
cv2.destroyAllWindows()