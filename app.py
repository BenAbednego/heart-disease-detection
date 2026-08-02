import gradio as gr
import pandas as pd
import numpy as np
import pickle

try:
    import spaces
except ImportError:
    spaces = None

# Load model & scalers
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler1.pkl', 'rb') as f:
    scaler1 = pickle.load(f)

with open('scaler2.pkl', 'rb') as f:
    scaler2 = pickle.load(f)

def predict_logic(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    sex_enc = 1 if sex == 'Pria' else 0
    chest_enc = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}[chest_pain]
    restingecg_enc = {'LVH': 0, 'Normal': 1, 'ST': 2}[resting_ecg]
    st_enc = {'Down': 0, 'Flat': 1, 'Up': 2}[st_slope]
    exercise_enc = 1 if exercise_angina == 'Ya' else 0

    sex_x_chest = sex_enc * chest_enc
    exercise_x_oldpeak = exercise_enc * float(oldpeak)
    maxhr_x_stslope = float(max_hr) * st_enc
    age_x_restingbp = float(age) * float(resting_bp)

    temp1 = pd.DataFrame([{
        'Age': float(age),
        'RestingBP': float(resting_bp),
        'Cholesterol': float(cholesterol),
        'MaxHR': float(max_hr),
        'Oldpeak': float(oldpeak),
        'ChestPainType_Encoded': chest_enc,
        'RestingECG_Encoded': restingecg_enc,
        'ST_Slope_Encoded': st_enc,
        'FastingBS': 1.0 if fasting_bs == 'Ya (>120 mg/dL)' else 0.0,
        'Sex_Encoded': sex_enc,
        'ExerciseAngina_Encoded': exercise_enc
    }])
    scaled1 = scaler1.transform(temp1)

    temp2 = pd.DataFrame([{
        'MaxHR_x_STSlope': maxhr_x_stslope,
        'Age_x_RestingBP': age_x_restingbp
    }])
    scaled2 = scaler2.transform(temp2)

    input_data = pd.DataFrame([{
        'Age_Scaled': scaled1[0][0],
        'RestingBP_Scaled': scaled1[0][1],
        'Cholesterol_Scaled': scaled1[0][2],
        'MaxHR_Scaled': scaled1[0][3],
        'Oldpeak_Scaled': scaled1[0][4],
        'ChestPainType_Encoded_Scaled': scaled1[0][5],
        'RestingECG_Encoded_Scaled': scaled1[0][6],
        'ST_Slope_Encoded_Scaled': scaled1[0][7],
        'FastingBS_Scaled': scaled1[0][8],
        'Sex_Encoded_Scaled': scaled1[0][9],
        'ExerciseAngina_Encoded_Scaled': scaled1[0][10],
        'Sex_x_ChestPain': sex_x_chest,
        'Exercise_x_Oldpeak': exercise_x_oldpeak,
        'MaxHR_x_STSlope_Scaled': scaled2[0][0],
        'Age_x_RestingBP_Scaled': scaled2[0][1]
    }])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    prob_sakit = round(float(prob[1]) * 100, 2)
    prob_sehat = round(float(prob[0]) * 100, 2)

    if pred == 1:
        return f"⚠️  BERISIKO PENYAKIT JANTUNG\n\nProbabilitas Sakit  : {prob_sakit}%\nProbabilitas Sehat  : {prob_sehat}%"
    else:
        return f"✅  TIDAK BERISIKO (SEHAT)\n\nProbabilitas Sehat  : {prob_sehat}%\nProbabilitas Sakit  : {prob_sakit}%"

if spaces:
    @spaces.GPU
    def predict_heart_disease(*args, **kwargs):
        return predict_logic(*args, **kwargs)
else:
    predict_heart_disease = predict_logic

demo = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Slider(18, 100, value=50, label="Usia (Age)"),
        gr.Radio(["Pria", "Wanita"], value="Pria", label="Jenis Kelamin (Sex)"),
        gr.Dropdown(["ASY", "ATA", "NAP", "TA"], value="ASY", label="Tipe Nyeri Dada (Chest Pain Type)"),
        gr.Slider(80, 200, value=120, label="Tekanan Darah Istirahat (Resting BP)"),
        gr.Slider(100, 600, value=220, label="Kolesterol (Cholesterol)"),
        gr.Radio(["Tidak (<=120 mg/dL)", "Ya (>120 mg/dL)"], value="Tidak (<=120 mg/dL)", label="Gula Darah Puasa (Fasting BS)"),
        gr.Dropdown(["Normal", "ST", "LVH"], value="Normal", label="Hasil EKG Istirahat (Resting ECG)"),
        gr.Slider(60, 220, value=150, label="Detak Jantung Maksimum (Max HR)"),
        gr.Radio(["Tidak", "Ya"], value="Tidak", label="Nyeri Dada Saat Olahraga (Exercise Angina)"),
        gr.Slider(0.0, 6.0, value=0.0, step=0.1, label="ST Depression (Oldpeak)"),
        gr.Dropdown(["Up", "Flat", "Down"], value="Flat", label="ST Slope")
    ],
    outputs=gr.Textbox(label="Hasil Diagnosis & Estimasi Risiko"),
    title="🫀 Prediksi Dini Penyakit Jantung (Heart Disease Early Detection)",
    description="Sistem deteksi dini penyakit jantung berbasis Machine Learning (XGBoost Classifier)."
)

if __name__ == '__main__':
    demo.launch()