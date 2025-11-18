from ultralytics import YOLO
from PIL import Image
import cv2
import easyocr
import numpy as np

model = YOLO('/Users/DELL/OneDrive/Desktop/noron/ALPR/ALPR/runs/detect/train2/weights/best.pt')

results = model('/Users/DELL/OneDrive/Desktop/noron/ALPR/ALPR/test2.jpeg')
model.export(format="onnx")

for r in results:
    print(r.boxes)
    im_array=r.plot()
    im = Image.fromarray(im_array[...,::-1])
    im.show()
    im.save('kq.jpg')