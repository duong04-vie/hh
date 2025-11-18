import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json

# ================== CẤU HÌNH ==================
train_dir = 'dataset_cnn/train'
val_dir   = 'dataset_cnn/val'
test_dir  = 'dataset_cnn/test'

img_height, img_width = 32, 32
batch_size = 32
epochs = 50

# Tạo thư mục lưu kết quả nếu chưa có
output_folder = 'training_results'
os.makedirs(output_folder, exist_ok=True)

# ================== LOAD DATA ==================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen   = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# Tự động lấy số lớp
num_classes = len(train_generator.class_indices)
print(f"Số lớp phát hiện: {num_classes}")
print("Danh sách lớp:", train_generator.class_indices)

# ================== XÂY DỰNG MODEL ==================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ================== TRAIN ==================
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# ================== ĐÁNH GIÁ ==================
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"\nĐộ chính xác trên tập test: {test_acc*100:.2f}%")

# ================== LƯU MODEL ==================
model.save('viet_lp_char_cnn.h5')
print("Model đã lưu: viet_lp_char_cnn.h5")

# ================== VẼ & LƯU BIỂU ĐỒ ==================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Biểu đồ Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
loss_path = os.path.join(output_folder, f'loss_curve_{timestamp}.png')
plt.savefig(loss_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Đã lưu biểu đồ Loss: {loss_path}")

# Biểu đồ Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
acc_path = os.path.join(output_folder, f'accuracy_curve_{timestamp}.png')
plt.savefig(acc_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Đã lưu biểu đồ Accuracy: {acc_path}")

# ================== LƯU THỨ TỰ LỚP ĐỂ DÙNG SAU NÀY ==================
class_indices = train_generator.class_indices
# Đảo ngược để có index → ký tự
idx_to_class = {v: k for k, v in class_indices.items()}

# Lưu ra file JSON
with open('class_labels.json', 'w') as f:
    json.dump(idx_to_class, f)

print("Đã lưu thứ tự lớp vào class_labels.json")
print("Thứ tự lớp (rất quan trọng):")
for idx, label in sorted(idx_to_class.items()):
    print(f"{idx}: {label}")

print("\nHOÀN TẤT! Tất cả file đã lưu trong thư mục:", output_folder)