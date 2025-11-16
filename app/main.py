import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pathlib 

# Configure all paths at the top - works on Windows, Linux, and macOS
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # Project root directory
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'model'
ASSETS_DIR = BASE_DIR / 'assets'

# Specific file paths
DATA_FILE = DATA_DIR / 'breast_cancer_data.csv'
MODEL_FILE = MODEL_DIR / 'model.pkl'
SCALER_FILE = MODEL_DIR / 'scaler.pkl'
CSS_FILE = ASSETS_DIR / 'style.css'

def get_clean_data():
    data = pd.read_csv(DATA_FILE)
    data.drop(columns=['Unnamed: 32','id'],axis=1,inplace=True)
    data.diagnosis = data.diagnosis.map({'B':0, 'M':1})
    return data

def add_slidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data = get_clean_data()

    slider_labels = [
    ("radius(mean)", "radius_mean"),
    ("texture(mean)", "texture_mean"),
    ("perimeter(mean)", "perimeter_mean"),
    ("area(mean)", "area_mean"),
    ("smoothness(mean)", "smoothness_mean"),
    ("compactness(mean)", "compactness_mean"),
    ("concavity(mean)", "concavity_mean"),
    ("concavepoints(mean)", "concave points_mean"),
    ("symmetry(mean)", "symmetry_mean"),
    ("fractaldimension(mean)", "fractal_dimension_mean"),
    ("radius(se)", "radius_se"),
    ("texture(se)", "texture_se"),
    ("perimeter(se)", "perimeter_se"),
    ("area(se)", "area_se"),
    ("smoothness(se)", "smoothness_se"),
    ("compactness(se)", "compactness_se"),
    ("concavity(se)", "concavity_se"),
    ("concavepoints(se)", "concave points_se"),
    ("symmetry(se)", "symmetry_se"),
    ("fractaldimension(se)", "fractal_dimension_se"),
    ("radius(worst)", "radius_worst"),
    ("texture(worst)", "texture_worst"),
    ("perimeter(worst)", "perimeter_worst"),
    ("area(worst)", "area_worst"),
    ("smoothness(worst)", "smoothness_worst"),
    ("compactness(worst)", "compactness_worst"),
    ("concavity(worst)", "concavity_worst"),
    ("concavepoints(worst)", "concave points_worst"),
    ("symmetry(worst)", "symmetry_worst"),
    ("fractaldimension(worst)", "fractal_dimension_worst")
]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value= float(data[key].mean())
        )

    return input_dict

def get_scaled_values(input_data):
    data = get_clean_data()

    X = data.drop(['diagnosis'],axis=1)

    scaled_data = {}

    for key, value in input_data.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value-min_val)/(max_val-min_val)
        scaled_data[key] = scaled_value

    return scaled_data

def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concavepoints",
    "symmetry",
    "fractaldimension"
]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
    input_data['radius_mean'],
    input_data['texture_mean'],
    input_data['perimeter_mean'],
    input_data['area_mean'],
    input_data['smoothness_mean'],
    input_data['compactness_mean'],
    input_data['concavity_mean'],
    input_data['concave points_mean'],
    input_data['symmetry_mean'],
    input_data['fractal_dimension_mean']
],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
    input_data['radius_se'],
    input_data['texture_se'],
    input_data['perimeter_se'],
    input_data['area_se'],
    input_data['smoothness_se'],
    input_data['compactness_se'],
    input_data['concavity_se'],
    input_data['concave points_se'],
    input_data['symmetry_se'],
    input_data['fractal_dimension_se']
],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'],
            input_data['texture_worst'],
            input_data['perimeter_worst'],
            input_data['area_worst'],
            input_data['smoothness_worst'],
            input_data['compactness_worst'],
            input_data['concavity_worst'],
            input_data['concave points_worst'],
            input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        height=600,
    )

    return fig

def add_predictions(input_data):
    model = pickle.load(open(MODEL_FILE, 'rb'))
    scaler = pickle.load(open(SCALER_FILE, 'rb'))

    input_array = np.array(list(input_data.values())).reshape(1,-1)

    input_array_scaled = scaler.transform(input_array)

    predictions = model.predict(input_array_scaled)

    st.subheader('Cell Cluster Prediction')
    st.write('The cell cluster  is : ')

    if predictions[0] == 0 :
        st.write('<span class="diagnosis-benign">Benign</span>',unsafe_allow_html=True)
    else:
        st.write('<span class="diagnosis-malignant">Malignant</span>',unsafe_allow_html=True)

    st.write(f"Probability of Benign:\n<span class='prediction'> {model.predict_proba(input_array_scaled)[0][0]:.2f}</span>",unsafe_allow_html=True)
    st.write(f"Probability of Malignant:\n<span class='prediction'> {model.predict_proba(input_array_scaled)[0][1]:.2f}</span>",unsafe_allow_html=True)

    st.write('''This result is generated by an AI model for educational purposes only.
It is not a substitute for a professional medical diagnosis.
Please consult a qualified healthcare provider for any health concerns or decisions.''')


def main():
    st.set_page_config(
        page_title='Breast Cancer Predictor',
        page_icon= ':female-doctor:',
        layout='wide',
        initial_sidebar_state='expanded',

    )

    with open(CSS_FILE, 'r', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    input_data = add_slidebar()
    # st.write(input_data)

    with st.container():
        st.title('Breast Cancer Predictor')
        st.write("Welcome to the Breast Cancer Predictor! Enter your medical data and let our AI suggest your risk level in seconds.")

    col1,col2 = st.columns([4,1])

    with col1:
        fig = get_radar_chart(input_data)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        add_predictions(input_data)



if __name__ == '__main__':
    main()