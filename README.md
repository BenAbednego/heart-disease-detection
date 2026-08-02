---
title: Heart Disease Early Detection
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
---

# 🫀 Heart Disease Early Detection System (Prediksi Dini Penyakit Jantung)

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble%20ML-111111?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/benabednego/heart-disease-detection)

Aplikasi Web dan Sistem Machine Learning **End-to-End** untuk melakukan **skrining dini risiko penyakit jantung** berdasarkan 11 indikator klinis pasien. Sistem ini ditenagai oleh model ensemble **XGBoost Classifier** yang telah dioptimalkan melalui *hyperparameter tuning* dan *feature engineering*.

---

## 📌 1. Insights & Temuan Utama Data (`heart.csv`)

### **a. Profil Dataset**
- **Ukuran Data**: 918 sampel pasien (setelah pembersihan: 917 baris valid).
- **Variabel Target (`HeartDisease`)**:
  - `0`: Sehat / Tidak Berisiko
  - `1`: Berisiko Sakit Jantung
- **Distribusi Target**: Balanced dataset (~55% berisiko penyakit jantung).

### **b. Indikator Klinis (Features)**
1. **`Age`**: Usia pasien (rentang 28 – 77 tahun). Pasien di atas usia 50 tahun menunjukkan korelasi lebih tinggi terhadap risiko penyakit jantung.
2. **`Sex`**: Jenis kelamin (`M` / `F`). Pria menunjukkan proporsi risiko yang lebih tinggi dalam dataset ini.
3. **`ChestPainType`**: Tipe nyeri dada:
   - `ASY` (Asymptomatic): Nyeri tanpa gejala khas — **faktor risiko tertinggi**.
   - `ATA` (Atypical Angina), `NAP` (Non-Anginal Pain), `TA` (Typical Angina).
4. **`RestingBP`**: Tekanan darah istirahat ($mmHg$). Nilai normal 90-120 mmHg.
5. **`Cholesterol`**: Kadar serum kolesterol ($mg/dL$). Ditemukan 172 data abnormal ($0\text{ mg/dL}$) yang berhasil diimputasi menggunakan median valid ($223\text{ mg/dL}$).
6. **`FastingBS`**: Gula darah puasa ($> 120\text{ mg/dL} \rightarrow 1$, else $0$).
7. **`RestingECG`**: Hasil elektrokardiogram istirahat (`Normal`, `ST`, `LVH`).
8. **`MaxHR`**: Detak jantung maksimum selama latihan fisik (60 – 202 bpm).
9. **`ExerciseAngina`**: Nyeri dada akibat aktivitas fisik (`Y` / `N`).
10. **`Oldpeak`**: Depresi ST akibat olahraga relatif terhadap istirahat (-2.6 s/d 6.2).
11. **`ST_Slope`**: Kemiringan segmen ST pada puncak latihan (`Up`, `Flat`, `Down`). Segmen `Flat` & `Down` mengindikasikan iskemia miokard tinggi.

---

## 🛠️ 2. Data Preprocessing & Feature Engineering

1. **Data Cleaning**:
   - Menghapus 1 baris tidak valid dengan `RestingBP == 0`.
   - Mengimputasi 172 baris dengan nilai `Cholesterol == 0` menggunakan nilai **median kolesterol valid** ($223\text{ mg/dL}$).
2. **Categorical Encoding**:
   - `LabelEncoder` diterapkan pada fitur kategorikal (`Sex`, `ChestPainType`, `RestingECG`, `ST_Slope`, `ExerciseAngina`).
3. **Feature Scaling**:
   - `MinMaxScaler` mengompresi rentang fitur ke interval $[0, 1]$.
4. **Feature Engineering (4 Interaksi Variabel Klinis)**:
   - `Sex_x_ChestPain`: Interaksi jenis kelamin & tipe nyeri dada.
   - `Exercise_x_Oldpeak`: Interaksi angina latihan fisik & depresi segmen ST (*oldpeak*).
   - `MaxHR_x_STSlope`: Interaksi detak jantung maks & kemiringan segmen ST.
   - `Age_x_RestingBP`: Interaksi umur & tekanan darah istirahat.

---

## 🤖 3. Performa & Perbandingan Model ML

Pembagian dataset: **80% Training Data (733 sampel)** dan **20% Test Data (184 sampel)**.

### **Hasil Evaluasi Model (Setelah Hyperparameter Tuning via GridSearchCV)**

| Model Machine Learning | Training Accuracy | Testing Accuracy | F1-Score | ROC-AUC | Overfitting Gap | Status Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **XGBoost Classifier** | **94.82%** | **88.04%** | **90.00%** | **93.07%** | **6.78%** | **Model Terbaik & Paling Stabil** |
| 🥈 **Gradient Boosting** | 95.91% | 85.33% | 87.78% | 92.47% | 10.58% | Cukup Baik |
| 🥉 **Random Forest** | 89.09% | 83.70% | 86.24% | 92.83% | 5.39% | Akurasi Turun |

> **Alasan Pemilihan XGBoost**:
> Model XGBoost mencapai **Akurasi 88.04%** dan **ROC-AUC 93.07%**. Nilai **F1-Score 90.00%** sangat krusial di dunia medis karena meminimalkan angka *False Negative* (pasien sakit yang terprediksi sehat).

---

## 🖥️ 4. Arsitektur Aplikasi Web

```
Client (Browser) <---> Flask REST API (app.py) <---> Scalers (scaler1, scaler2) <---> XGBoost Model (.pkl)
```

- **Backend Framework**: Python Flask (`app.py`)
- **Frontend UI**: Responsive Vanilla HTML5 + CSS3 (Modern Dark Glassmorphism, Google Fonts Inter).
- **API Endpoint**: `POST /prediksi`
  - Input JSON: 11 Parameter Kesehatan Pasien.
  - Output JSON: `prediksi` (`0`/`1`), `prob_sehat` (%), `prob_sakit` (%).

---

## 📁 5. Struktur Direktori Proyek

```
Heart Deasese/
├── README.md                         # Dokumentasi lengkap & konfigurasi Hugging Face
├── Dockerfile                        # Konfigurasi containerisasi Docker
├── app.py                            # Server Flask utama (Backend & Route)
├── index.html                        # Tampilan antarmuka Glassmorphism UI
├── heart.csv                         # Dataset klinis penyakit jantung
├── requirements.txt                  # Library & dependensi Python
├── xgboost_model.pkl                 # Model terlatih XGBoost
├── scaler1.pkl                       # Scaler 1 (11 fitur utama)
├── scaler2.pkl                       # Scaler 2 (fitur interaksi)
├── ipynb/                            # Folder riset & Jupyter Notebook
│   └── Hospital_Project.ipynb        # Experiment Notebook (EDA, Modeling, Evaluation)
└── HeartDisease_WebApp/              # Folder modul web pendukung
```

---

## 🚀 6. Panduan Menjalankan Lokal

### **1. Clone Repository & Masuk ke Folder**
```bash
git clone https://github.com/BenAbednego/heart-disease-detection.git
cd heart-disease-detection
```

### **2. Buat & Aktifkan Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows
```

### **3. Install Dependensi**
```bash
pip install -r requirements.txt
```

### **4. Jalankan Aplikasi**
```bash
python app.py
```
Buka browser di: **`http://localhost:7860`** atau **`http://127.0.0.1:7860`**.

---

## 🌐 7. Deployment Status

- **Hugging Face Spaces**: [https://huggingface.co/spaces/benabednego/heart-disease-detection](https://huggingface.co/spaces/benabednego/heart-disease-detection)
- **GitHub Repository**: [https://github.com/BenAbednego/heart-disease-detection](https://github.com/BenAbednego/heart-disease-detection)

---

### 📜 Lisensi
Dikembangkan untuk keperluan riset dan aplikasi teknologi informasi kesehatan (MIT License).
