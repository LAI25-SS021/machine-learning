import tensorflow as tf
import numpy as np
import cv2
import sys

# Load model
model = tf.keras.models.load_model("models/best_model.keras")

# Label klasifikasi kulit wajah
labels = ["Berminyak", "Kering", "Normal"]


# Fungsi preprocessing gambar
def preprocess_image(image_path, target_size=(300, 300)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di path: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Ubah ke shape (1, height, width, channels)
    return img


# Fungsi prediksi
def predict(image_path):
    try:
        image = preprocess_image(image_path)
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)
        confidence = float(np.max(predictions))
        print(f"Prediksi: {labels[predicted_class]} (Confidence: {confidence:.2f})")
    except Exception as e:
        print("Terjadi kesalahan:", str(e))


# Jalankan dari CLI
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py path/to/image.jpg")
    else:
        predict(sys.argv[1])
