from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'ulcer_model.pkl'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ulcer-predictor-backend.vercel.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)


class Patient(BaseModel):
    gender: str
    age_group: str
    burning_pain: bool
    pain_empty_stomach: bool
    pain_after_eating: bool
    night_pain: bool
    bloating: bool
    nausea: bool
    heartburn: bool
    chest_pain: bool
    swallowing_difficulty: bool
    appetite_loss: bool
    black_stool: bool
    vomiting_blood: bool
    stress: bool
    nsaid_use: bool
    h_pylori: bool


gender_map = {'Female': 0, 'Male': 1}

age_map = {
    'Teen(13-19)': 1,
    'Young Adult(20-39)': 2,
    'Middle Age(40-59)': 3,
    'Senior(60+)': 4,
}


@app.get('/')
def home():
    return {'status': 'API Running'}


@app.post('/predict')
def predict(patient: Patient):
    df = pd.DataFrame(
        [[
            gender_map[patient.gender],
            int(patient.burning_pain),
            int(patient.pain_empty_stomach),
            int(patient.pain_after_eating),
            int(patient.night_pain),
            int(patient.bloating),
            int(patient.nausea),
            int(patient.heartburn),
            int(patient.chest_pain),
            int(patient.swallowing_difficulty),
            int(patient.appetite_loss),
            int(patient.black_stool),
            int(patient.vomiting_blood),
            int(patient.stress),
            int(patient.nsaid_use),
            int(patient.h_pylori),
            age_map[patient.age_group],
        ]],
        columns=[
            'gender',
            'burning_pain',
            'pain_empty_stomach',
            'pain_after_eating',
            'night_pain',
            'bloating',
            'nausea',
            'heartburn',
            'chest_pain',
            'swallowing_difficulty',
            'appetite_loss',
            'black_stool',
            'vomiting_blood',
            'stress',
            'nsaid_use',
            'h_pylori',
            'age_group',
        ],
    )

    risk = float(model.predict(df)[0])
    risk = max(0, min(100, risk))

    if risk < 30:
        level = 'Low Risk'
    elif risk < 60:
        level = 'Moderate Risk'
    elif risk < 80:
        level = 'High Risk'
    else:
        level = 'Very High Risk'

    ulcer_type = 'No Clear Ulcer Pattern'

    if patient.chest_pain and patient.swallowing_difficulty:
        ulcer_type = 'Esophageal Ulcer'
    elif patient.pain_empty_stomach and patient.night_pain:
        ulcer_type = 'Duodenal Ulcer'
    elif patient.pain_after_eating and patient.burning_pain:
        ulcer_type = 'Gastric Ulcer'

    suggestions = []

    if patient.stress:
        suggestions.append('Reduce stress through proper sleep and exercise.')
    if patient.nsaid_use:
        suggestions.append('Avoid excessive painkiller use.')
    if patient.h_pylori:
        suggestions.append('Consult a doctor for H. pylori treatment.')
    if patient.heartburn:
        suggestions.append('Avoid spicy and oily foods.')
    if patient.bloating:
        suggestions.append('Eat smaller meals regularly.')
    if patient.burning_pain:
        suggestions.append('Avoid skipping meals.')
    if patient.nausea:
        suggestions.append('Drink plenty of water.')
    if patient.black_stool or patient.vomiting_blood:
        suggestions.append('Seek immediate medical attention.')

    if not suggestions:
        suggestions.append('Maintain a healthy lifestyle.')

    return {
        'risk': round(risk, 2),
        'level': level,
        'ulcer_type': ulcer_type,
        'suggestions': suggestions,
    }
