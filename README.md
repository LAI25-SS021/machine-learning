# Machine Learning Ropakoe : AI Deteksi Tipe Kulit Wajah

Jupyter Notebook ini berisi pipeline lengkap untuk membangun model deteksi tipe kulit wajah menggunakan arsitektur MobileNetV2. Proyek ini bertujuan untuk mengklasifikasikan gambar wajah ke dalam tiga kategori: **berminyak**, **kering**, dan **normal**.

## Sumber Data

Kaggle : `https://www.kaggle.com/datasets/igecko/klasifikasi-kulit-wajah-roboflow-ebn3-vers`
Dataset tersebut telah diunduh dan disimpan di Google Drive. Link akses dataset dapat ditemukan pada bagian awal notebook.

## Fitur Utama

- **Data Preparation**:

  - Download dataset dari Google Drive.
  - Filter kelas yang diinginkan dan struktur ulang direktori.
  - Visualisasi distribusi data dan resolusi gambar.

- **Data Preprocessing**:

  - Normalisasi piksel gambar.
  - Resize gambar ke ukuran 300x300 piksel.
  - Split data menjadi train, validation, dan test.

- **Modeling**:

  - Implementasi tiga versi model MobileNetV2 dengan variasi arsitektur dan augmentasi data.
  - Callback custom untuk early stopping berdasarkan akurasi.
  - Training, evaluasi, dan penyimpanan model serta history.

- **Evaluasi & Visualisasi**:

  - Perbandingan performa model (akurasi, loss).
  - Plot training/validation accuracy & loss.
  - Evaluasi pada data test (classification report & confusion matrix).

- **Inference**:
  - Fungsi prediksi gambar baru dengan visualisasi hasil dan confidence score.

## Cara Menggunakan

1. **Persiapan Lingkungan**

   - Pastikan telah menginstal dependensi: TensorFlow, Keras, scikit-image, pandas, matplotlib, seaborn, tqdm, gdown, dan Google Colab (jika menggunakan Colab).

2. **Jalankan Notebook**

   - Ikuti setiap cell secara berurutan mulai dari import library, data preparation, hingga inference.

3. **Training Model**

   - Tiga model MobileNetV2 akan dilatih dengan konfigurasi berbeda. Model terbaik dipilih berdasarkan akurasi pada data test.

4. **Evaluasi & Inference**
   - Lakukan evaluasi performa model dan prediksi gambar baru dengan fungsi yang telah disediakan.

## Struktur Model

- **MobileNetV2 v1**: Dense(64) + Dropout(0.5)
- **MobileNetV2 v2**: Dropout(0.3) + Dense(64) + Dropout(0.3)
- **MobileNetV2 v3**: Sama seperti v2, namun dengan augmentasi data tambahan

## Output

- Model terlatih disimpan dalam format `.keras` dan `.h5`.
- History training disimpan dalam file `.json`.
- Notebook menghasilkan visualisasi distribusi data, training curves, confusion matrix, dan classification report.

## Catatan

- Notebook ini dioptimalkan untuk dijalankan di Google Colab.
- Dataset harus tersedia di Google Drive dan dapat diakses melalui link yang disediakan.
- Untuk inference, upload gambar melalui Colab dan jalankan cell prediksi.
- Model terbaik akan disimpan juga dalam format '.h5' dan 'saved model'.

---

**Penulis:**  
Tim LAI25-SS021 - Capstone Project 2025
