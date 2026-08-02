from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler1.pkl', 'rb') as f:
    scaler1 = pickle.load(f)

with open('scaler2.pkl', 'rb') as f:
    scaler2 = pickle.load(f)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/prediksi', methods=['POST'])
def prediksi():
    data = request.json

    age = float(data['age'])
    sex = data['sex']
    chest = data['chest']
    restingbp = float(data['restingbp'])
    cholesterol = float(data['cholesterol'])
    fasting = float(data['fasting'])
    restingecg = data['restingecg']
    maxhr = float(data['maxhr'])
    exercise = data['exercise']
    oldpeak = float(data['oldpeak'])
    st = data['st']

    sex_enc = 1 if sex == 'M' else 0
    chest_enc = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}[chest]
    restingecg_enc = {'LVH': 0, 'Normal': 1, 'ST': 2}[restingecg]
    st_enc = {'Down': 0, 'Flat': 1, 'Up': 2}[st]
    exercise_enc = 1 if exercise == 'Y' else 0

    sex_x_chest = sex_enc * chest_enc
    exercise_x_oldpeak = exercise_enc * oldpeak
    maxhr_x_stslope = maxhr * st_enc
    age_x_restingbp = age * restingbp

    # Scaling pakai scaler1
    temp1 = pd.DataFrame([{
        'Age': age,
        'RestingBP': restingbp,
        'Cholesterol': cholesterol,
        'MaxHR': maxhr,
        'Oldpeak': oldpeak,
        'ChestPainType_Encoded': chest_enc,
        'RestingECG_Encoded': restingecg_enc,
        'ST_Slope_Encoded': st_enc,
        'FastingBS': fasting,
        'Sex_Encoded': sex_enc,
        'ExerciseAngina_Encoded': exercise_enc
    }])
    scaled1 = scaler1.transform(temp1)

    # Scaling pakai scaler2
    temp2 = pd.DataFrame([{
        'MaxHR_x_STSlope': maxhr_x_stslope,
        'Age_x_RestingBP': age_x_restingbp
    }])
    scaled2 = scaler2.transform(temp2)

    # Input final
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

    hasil = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    return jsonify({
        'prediksi': int(hasil),
        'prob_sehat': round(float(prob[0]) * 100, 2),
        'prob_sakit': round(float(prob[1]) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)