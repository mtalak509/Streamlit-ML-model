import pandas as pd
import numpy as np
import streamlit as st

import sklearn
sklearn.set_config(transform_output="pandas")

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, TargetEncoder
from sklearn.model_selection import GridSearchCV, KFold
import joblib


model = joblib.load('RF_ml_pipeline.pkl')
preprocessor = joblib.load('Preprocessor_pipeline.pkl')


# Настройка заголовка приложения
st.title("Сбор медицинских данных")

# Создание боковой панели для ввода данных
with st.sidebar:
    st.header("Введите информацию")
    
    # Поля ввода
    age = st.number_input("Age (возраст пациента)", min_value=1, max_value=120, value=30)
    
    # Выпадающий список для пола
    sex = st.selectbox("Sex (пол)", ["M (Мужской)", "F (Женский)"])
    
    # Выпадающий список для типа боли в груди
    chest_pain = st.selectbox(
        "ChestPainType (тип боли в груди)",
        ["TA", "ATA", "NAP", "ASY"]
    )
    
    resting_bp = st.number_input("RestingBP (артериальное давление покоя, мм рт.ст.)", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol (холестерин, мг/дл)", min_value=100, max_value=600, value=200)
    
    # Выпадающий список для FastingBS
    fasting_bs = st.selectbox("FastingBS (сахар в крови)", [0, 1])
    
    # Выпадающий список для ЭКГ
    resting_ecg = st.selectbox(
        "RestingECG (результаты ЭКГ)",
        ["Normal", "ST", "LVH"]
    )
    
    max_hr = st.number_input("MaxHR (максимальная ЧСС)", min_value=60, max_value=202, value=150)
    
    # Выпадающий список для ExerciseAngina
    exercise_angina = st.selectbox("ExerciseAngina (ангина при нагрузке)", ["Y (Да)", "N (Нет)"])
    
    oldpeak = st.number_input("Oldpeak (депрессия ST)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    
    # Выпадающий список для ST_Slope
    st_slope = st.selectbox(
        "ST_Slope (наклон ST)",
        ["Up", "Flat", "Down"]
    )
    
    # Кнопка для добавления данных
    add_data = st.button("Добавить запись")

# Основная часть приложения
st.header("Собранные данные")

# Инициализация пустого датафрейма
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(
        columns=[
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
        ]
    )

# Добавление новых данных при нажатии кнопки
if add_data:
    new_row = {
        'Age': age,
        'Sex': sex[0],  # Берем только первую букву (M/F)
        'ChestPainType': chest_pain,  
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg, 
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina[0], 
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    
    # Добавление новой строки в датафрейм
    st.session_state.df = pd.concat(
        [st.session_state.df, pd.DataFrame([new_row])],
        ignore_index=True
    )
    
    st.success("Данные успешно добавлены!")

# Отображение дата


data = st.session_state.df
st.write(data)

try:
    preprocessed_data = preprocessor.transform(data)
    res = model.predict(preprocessed_data)
    if res == 1:
        st.header("С вероятностью 88% у Вас обнаружится порок сердца")
    else:
        st.header("С вероятностью 88% у Вас не обнаружится порок сердца")
      
except:
    st.write('Данные не загружены. Внесите свои данные')
 