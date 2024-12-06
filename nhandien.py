import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os

# Bước 1: Chuẩn bị dữ liệu
def prepare_data():
    train_data_dir = 'D:/anhtest/train'
    validation_data_dir = 'D:/anhtest/test'

    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale'
    )

    return train_generator, validation_generator

# Bước 2: Xây dựng mô hình CNN
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # 7 cảm xúc

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Bước 3: Huấn luyện mô hình
def train_model():
    train_generator, validation_generator = prepare_data()  # Lấy dữ liệu
    model = build_model()  # Xây dựng mô hình

    # Kiểm tra xem mô hình đã tồn tại chưa
    if os.path.exists('emotion_detection.h5'):
        model = load_model('emotion_detection.h5')  # Tải mô hình đã lưu
        print("Mô hình đã được tải thành công.")
    else:
        history = model.fit(train_generator, validation_data=validation_generator, epochs=50)
        model.save('emotion_detection.h5')  # Lưu mô hình

        # Lưu lịch sử huấn luyện
        history_df = pd.DataFrame(history.history)
        history_df.to_csv('training_history.csv', index=False)  # Lưu vào tệp CSV
        print("Mô hình đã được huấn luyện và lưu thành công.")

# Bước 4: Nhận diện cảm xúc từ video
def detect_emotion():
    global cap  # Để có thể truy cập biến cap từ hàm khác
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    classifier = load_model('emotion_detection.h5')
    class_labels = ['Giận dữ', 'Ghê sợ', 'Sợ hãi', 'Hạnh phúc', 'Buồn', 'Bất ngờ', 'Trung lập']
    font = ImageFont.truetype("./arial.ttf", 32)
    cap = cv2.VideoCapture(0)

    # Tạo cửa sổ nhận diện cảm xúc
    emotion_window = tk.Toplevel(root)
    emotion_window.title("Nhận diện cảm xúc")    
    emotion_window.geometry("300x150")

    # Nút để thoát chương trình
    exit_button = tk.Button(emotion_window, text="Tắt", command=exit_program)
    exit_button.pack(pady=20)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm để thoát chương trình
def exit_program():
    # Giải phóng camera nếu đang mở
    if 'cap' in globals() and cap.isOpened():
        cap.release()
    root.destroy()  # Đóng cửa sổ chính
    cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ OpenCV

# Hàm để bắt đầu nhận diện cảm xúc
def start_emotion_detection():
    train_model()  # Huấn luyện mô hình
    detect_emotion()  # Nhận diện cảm xúc

# Hàm để xử lý phím tắt
def on_key_press(event):
    if event.keysym == 'Escape':  # Nếu phím Esc được nhấn
        exit_program()

# Chạy chương trình
if __name__ == "__main__":
    # Tạo giao diện Tkinter
    root = tk.Tk()
    root.title("NHAN DIEN CAM XUC")
    root.geometry("400x300")  # Thay đổi kích thước cửa sổ
    root.configure(bg='lightblue')  # Thay đổi màu nền

    # Thêm nhãn tiêu đề
    title_label = tk.Label(root, text="Nhận diện cảm xúc", font=("Arial", 20), bg='lightblue')
    title_label.pack(pady=20)

    # Nút để bắt đầu nhận diện cảm xúc
    start_button = tk.Button(root, text="Bắt đầu Nhận diện Cảm xúc", command=start_emotion_detection, font=("Arial", 14), bg='white', fg='blue')
    start_button.pack(pady=20)

    # Nút thoát
    exit_button = tk.Button(root, text="Thoát", command=exit_program, font=("Arial", 14), bg='red', fg='white')
    exit_button.pack(pady=20)

    # Gán sự kiện bàn phím
    root.bind('<KeyPress>', on_key_press)

    # Gán hàm exit_program cho sự kiện đóng cửa sổ
    root.protocol("WM_DELETE_WINDOW", exit_program)

    root.mainloop()