import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import load_model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, Lambda
from sklearn.preprocessing import OneHotEncoder
# Load your trained models
models = {
    'Random Forest Classifier': "grid_search_rf.pkl",
    "KNeighbors Classifier": "grid_search_knn.pkl",
    "Logistic Regression": "grid_search_lr.pkl",
    "XGBoost Classifier": "grid_search_xgb.pkl",
    "Convolutional Neural Network": "cnn.h5"
}
label_encoder_gender = joblib.load('label_encoder_gender.joblib')
label_encoder_work_type = joblib.load('label_encoder_work_type.joblib')
label_encoder_ever_married = joblib.load('label_encoder_ever_married.joblib')
label_encoder_smoking_status = joblib.load('label_encoder_smoking_status.joblib')
scaler = joblib.load('scaler.joblib')

# Load models from files
loaded_models = {name: joblib.load(filename) for name, filename in models.items() if name != "Convolutional Neural Network"}

# Function to handle any custom objects in the Lambda layer
def conditional_max_pooling(x):
    if x.shape[1] >= 2:
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.5)(x)
    return x

custom_objects = {
    'MaxPooling1D': MaxPooling1D,
    'Dropout': Dropout,
    'conditional_max_pooling': Lambda(conditional_max_pooling)
}

# Load CNN model
cnn_model = load_model("cnn.h5", custom_objects=custom_objects, compile=False, safe_mode= False)


st.set_page_config(page_title="Intelligent Stroke Predictor", page_icon=":tada:")

# ---- HEADER SECTION ----
with st.container():
    temp_bmi = 0
    temp_glucose = 0

    # st.subheader("WADDUP FUCKERS :wave:")
    st.title("Intelligent Stroke Predictor")
    st.subheader(
        "It never hurts to check..."
    )


# ---- About us
st.title('About')
st.header('', divider='red')

st.markdown("""
This app provides predictions on the likelihood of having a stroke based on user inputs. It leverages multiple machine learning models to offer accurate predictions and health recommendations.

Knowledge is power when it comes to your health. We believe that understanding your individual risk factors for stroke is the first step toward a healthier future. Our mission is to empower you with knowledge and awareness, enabling you to take proactive steps to reduce your risk of stroke.

Stroke is a serious medical condition, but many of its risk factors are manageable through lifestyle changes and early intervention. By providing us with some basic personal information, you can gain valuable insights into your unique risk profile. Our user-friendly tool will analyze your data and provide you with personalized recommendations to reduce your risk of stroke.

Remember, this tool is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a healthcare professional for a comprehensive assessment. However, our Stroke Risk Checker can serve as a useful starting point to help you take control of your health and well-being.

Your health matters, and taking proactive steps today can make a significant difference in your future. Let's work together to reduce the risk of stroke and promote a healthier, happier life. Get started now and take the first step towards a stroke-free tomorrow.





Personal data will NOT be collected.
"""
            )

st.info('''
**Key Features:**
- **User Input:** Easily input your health data and receive an instant analysis.
- **Risk Categorization:** Understand your risk level (Low, Medium, High) for stroke.
- **Health Recommendations:** Get personalized health tips to manage your health better.
- **Model Interpretability:** View feature importance to understand what factors influence your risk.

**How to Use:**
1. Enter your details in the sidebar.
2. Click the "Submit" button to get predictions and recommendations.
3. Follow the health tips provided to maintain or improve your health.
''')

# Streamlit app
st.sidebar.header("User Input Features")

def user_input_features():
    default_gender = "Male"
    default_age = 67
    default_hypertension = 0
    default_heart_disease = 0
    default_ever_married = "No"
    default_work_type = "Private"
    default_avg_glucose_level = 120.0
    default_bmi = 20.0
    default_smoking_status = "never smoked"
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"), index=("Male", "Female").index(default_gender))
    age = st.sidebar.slider("Age", 30, 100, default_age)
    hypertension = st.sidebar.selectbox("Hypertension", (0, 1), index=(0, 1).index(default_hypertension))
    heart_disease = st.sidebar.selectbox("Heart Disease", (0, 1), index=(0, 1).index(default_heart_disease))
    ever_married = st.sidebar.selectbox("Ever Married", ("Yes", "No"), index=("Yes", "No").index(default_ever_married))
    work_type = st.sidebar.selectbox("Work Type", ("Private", "Self-employed", "Govt_job"), 
                                     index=("Private", "Self-employed", "Govt_job").index(default_work_type))
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", 10.0, 400.0, default_avg_glucose_level)
    bmi = st.sidebar.slider("BMI", 10.0, 100.0, default_bmi)
    smoking_status = st.sidebar.selectbox("Smoking Status", ("formerly smoked", "never smoked", "smokes"), 
                                          index=("formerly smoked", "never smoked", "smokes").index(default_smoking_status))
    data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    
    features = pd.DataFrame(data, index=[0])
    return data

input_df = user_input_features()
features = pd.DataFrame(input_df, index=[0])

st.subheader("User Input Features")
st.write(features)

# Function to preprocess user input
df = features

df['gender'] = label_encoder_gender.transform(df['gender'])
df['work_type'] = label_encoder_work_type.transform(df['work_type'])
df['ever_married'] = label_encoder_ever_married.transform(df['ever_married'])
df['smoking_status'] = label_encoder_smoking_status.transform(df['smoking_status'])
numerical_features = ['age', 'avg_glucose_level', 'bmi']
df[numerical_features] = scaler.transform(df[numerical_features])

df['hypertension'] = df['hypertension']
df['heart_disease'] = df['heart_disease']
processed_df = df
rdf=df
# Model selection
model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))

# Add a submit button
if st.sidebar.button('Submit'):
    if model_choice != "Convolutional Neural Network":
        model = loaded_models[model_choice]
        proba = model.predict_proba(processed_df)
        #st.write(f"{model_choice} - Raw Probabilities: {proba}")
        prediction = proba[0][1]  # Probability of class 1 (Stroke)
        probabilities = proba[0]  # Probabilities for all classes
    else:
        cnn_scale = joblib.load('norm.joblib')
        processed_df = cnn_scale.transform(processed_df)
        if isinstance(processed_df, pd.DataFrame):
            processed_df = processed_df.values
        cnn_input = processed_df.reshape(processed_df.shape[0], processed_df.shape[1], 1)
        cnn_proba = cnn_model.predict(cnn_input)
        prediction = cnn_proba[0][0]
        probabilities = [1 - cnn_proba[0][0], cnn_proba[0][0]]

    # Risk categorization thresholds
    def categorize_risk(prob):
        if prob < 0.3:
            return "Low Risk"
        elif prob < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"

    # Health recommendations based on user input
    def health_recommendations(input_data):
        tips = []
        if input_data['hypertension'] == 1:
            tips.append("Manage your blood pressure through a healthy diet, regular exercise, and medication if prescribed.")
        if input_data['heart_disease'] == 1:
            tips.append("Monitor your heart health and consult with your doctor regularly.")
        if input_data['bmi'] > 25:
            tips.append("Maintain a healthy weight through balanced nutrition and physical activity.")
        if input_data['avg_glucose_level'] > 140:
            tips.append("Control your blood sugar levels with a proper diet and medication if necessary.")
        if input_data['smoking_status'] != "never smoked":
            tips.append("Consider quitting smoking to improve your overall health.")
        
        # Encouragement for those with normal inputs
        if input_data['hypertension'] == 0 and input_data['heart_disease'] == 0 and input_data['bmi'] <= 25 and input_data['avg_glucose_level'] <= 140 and input_data['smoking_status'] == "never smoked":
            tips.append("Great job! Continue with your healthy lifestyle and regular check-ups.")
            tips.append("Keep maintaining a balanced diet rich in fruits, vegetables, and whole grains.")
            tips.append("Stay active with regular physical exercise, at least 30 minutes a day.")
            tips.append("Ensure you get enough sleep and manage stress effectively.")
        
        return tips

    # Display predictions, probabilities, and recommendations
    st.subheader('Predictions, Probabilities, and Recommendations')
    risk_category = categorize_risk(prediction)
    st.write(f"{model_choice} - Probability of Stroke: {prediction:.2f} ({risk_category})")
    st.write(f"{model_choice} - Probability of No Stroke: {probabilities[0]:.2f}")
    
    if risk_category != "Low Risk" or True:  # Always show tips
        st.write("**Health Tips:**")
        tips = health_recommendations(input_df)
        for tip in tips:
            st.write(f"- {tip}")

    st.subheader('Feature Importance ')
    rf_model = joblib.load('best_Random Forest Classifier_model.pkl')
    importances = rf_model.feature_importances_
    features = rdf.columns
    feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
    st.bar_chart(feature_importance_df.set_index('feature'))

data_path = "health_data.csv"
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

all_data = load_data(data_path)

st.header('', divider='grey')
# st.title('Dataset Summary')
# st.subheader('Overview')
# st.write(data.head())

# st.subheader('Descriptive Statistics')
# st.write(data.describe())

# st.subheader('Missing Values')
# missing_values = data.isnull().sum()
# st.write(missing_values)
# Add a section for Model Information and Transparency
st.sidebar.title('Model Information and Transparency')
st.sidebar.subheader('Logistic Regression Metrics')
st.sidebar.write('''
- **Accuracy:** 74%
- **Recall:**  77%
- **Precision:** 73%
- **F1 Score:** 75%
- **MCC:** 49%
- **ROC AUC:** 80%
''')
# Information about Random Forest
st.sidebar.subheader('Random Forest Metrics')
st.sidebar.write('''
- **Accuracy:** 82%
- **Recall:**  87%
- **Precision:** 79%
- **F1 Score:** 83%
- **MCC:** 64%
- **ROC AUC:** 89%
''')

# Information about XGBoost
st.sidebar.subheader('XGBoost Metrics')
st.sidebar.write('''
- **Accuracy:** 81%
- **Recall:** 84%
- **Precision:** 79%
- **F1 Score:** 81%
- **MCC:** 61%
- **ROC AUC:** 83%
''')

# Information about KNN
st.sidebar.subheader('K-Nearest Neighbors (KNN) Metrics')
st.sidebar.write('''
**Mean Test Metrics:**
- **Accuracy:** 80%
- **Recall:** 90%
- **Precision:** 75%
- **F1 Score:** 82%
- **MCC:** 62%
- **ROC AUC:** 89%
''')

# Discussion section
st.sidebar.subheader("**CNN Metrics:**")
st.sidebar.write('''- Test Accuracy: 85%
- Precision: 84%
- Recall: 89%
- F1 Score: 87%
- MCC: 70%
- ROC AUC Score: 89%
''')
