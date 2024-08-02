import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import load_model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, Lambda
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

# Set up Streamlit
st.title('Stroke Prediction App')

st.title('About')
st.info('''
### Stroke Prediction App

This app provides predictions on the likelihood of having a stroke based on user inputs. It leverages multiple machine learning models to offer accurate predictions and health recommendations.

**Key Features:**
- **User Input:** Easily input your health data and receive an instant analysis.
- **Risk Categorization:** Understand your risk level (Low, Medium, High) for stroke.
- **Health Recommendations:** Get personalized health tips to manage your health better.
- **Model Interpretability:** View feature importance to understand what factors influence your risk.

**How to Use:**
1. Enter your details in the sidebar.
2. Click the "Submit" button to get predictions and recommendations.
3. Follow the health tips provided to maintain or improve your health.

**Disclaimer:**
This app is for informational purposes only and should not be used as a substitute for professional medical advice. Always consult with a healthcare provider for any health concerns.

Developed and powered by machine learning models to help you stay informed and healthy.
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
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 300.0, default_avg_glucose_level)
    bmi = st.sidebar.slider("BMI", 10.0, 70.0, default_bmi)
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
        st.write(f"{model_choice} - Raw Probabilities: {proba}")
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

    st.subheader('Feature Importance (Random Forest)')
    rf_model = joblib.load('best_Random Forest Classifier_model.pkl')
    importances = rf_model.feature_importances_
    features = rdf.columns
    feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
    st.bar_chart(feature_importance_df.set_index('feature'))

data_path = "health_data.csv"
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data = load_data(data_path)

# Display dataset summary
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

# Information about Logistic Regression
st.sidebar.subheader('Logistic Regression')
st.sidebar.write('''
**Parameters:**
- **C:** 1.5 (moderate regularization)
- **max_iter:** 100
- **solver:** saga

**Best Score:** 0.7435

**Mean Test Metrics:**
- **Accuracy:** 0.7435
- **Recall:** 0.7657
- **Precision:** 0.7342
- **F1 Score:** 0.7490
- **MCC:** 0.4883
- **ROC AUC:** 0.7994
''')
# Information about Random Forest
st.sidebar.subheader('Random Forest')
st.sidebar.write('''
**Parameters:**
- **max_depth:** 30
- **min_samples_leaf:** 1
- **min_samples_split:** 2
- **n_estimators:** 50

**Best Score:** 0.8183

**Mean Test Metrics:**
- **Accuracy:** 0.8183
- **Recall:** 0.8706
- **Precision:** 0.7942
- **F1 Score:** 0.8282
- **MCC:** 0.6436
- **ROC AUC:** 0.8878
''')

# Information about XGBoost
st.sidebar.subheader('XGBoost')
st.sidebar.write('''
**Parameters:**
- **colsample_bytree:** 1.0
- **learning_rate:** 0.2
- **max_depth:** 10
- **n_estimators:** 100
- **subsample:** 1.0

**Best Score:** 0.8058

**Mean Test Metrics:**
- **Accuracy:** 0.8058
- **Recall:** 0.8409
- **Precision:** 0.7911
- **F1 Score:** 0.8131
- **MCC:** 0.6163
- **ROC AUC:** 0.8344
''')

# Information about KNN
st.sidebar.subheader('K-Nearest Neighbors (KNN)')
st.sidebar.write('''
**Parameters:**
- **metric:** Manhattan
- **n_neighbors:** 5
- **weights:** Distance

**Best Score:** 0.8034

**Mean Test Metrics:**
- **Accuracy:** 0.8034
- **Recall:** 0.9052
- **Precision:** 0.7544
- **F1 Score:** 0.8218
- **MCC:** 0.6219
- **ROC AUC:** 0.8876
''')

# Discussion section
st.sidebar.title('Model Performance Discussion')
st.sidebar.write('''
**Performance Comparison:**
- The Random Forest model achieved the highest overall performance across the metrics evaluated, with notably strong recall and ROC AUC, indicating its robustness in identifying positive cases.
- Logistic Regression, while slightly lower in accuracy, maintained solid performance metrics and was consistent throughout various handling techniques for class imbalance.
- The XGBoost model performed well but did not surpass the Random Forest's metrics. However, it may still be preferred in scenarios requiring faster predictions or with larger datasets due to its efficient handling of feature importance.
- KNN model showed high recall but slightly lower precision than the XGBoost and Random Forest model, which could be indicative of potential overfitting on the minority class, making it less reliable for applications requiring a balanced approach.

**Performance Comparison:**
- The Random Forest model achieved the highest accuracy (0.8183) and F1 score (0.8282), indicating strong overall performance.
- The XGBoost model also performed well, with slightly lower accuracy (0.8058) and F1 score (0.8131) compared to Random Forest.
- KNN showed impressive recall (0.9052) but slightly lower accuracy (0.8034) and precision (0.7544), leading to a strong F1 score (0.8218).
- Logistic Regression had the lowest scores among the models, with an accuracy of 0.7435 and an F1 score of 0.7490.

**Metric Highlights:**
- Recall was highest for KNN (0.9052), indicating that it effectively identified positive instances.
- Precision was highest for Random Forest (0.7942), suggesting it had fewer false positives compared to the other models.
- Matthews Correlation Coefficient (MCC), which measures the quality of binary classifications, was highest for Random Forest (0.6436), followed by XGBoost (0.6163).

**Model Selection:**
- Based on overall performance, Random Forest appears to be the best model for this dataset, balancing high accuracy, precision, recall, and F1 score.
- XGBoost is a strong contender and might be preferred if interpretability and feature importance are critical factors.
- KNN is notable for its high recall, making it suitable for applications where identifying positive cases is crucial.

**Data Handling:**
- Imbalance Handling: Applied SMOTE to balance the dataset.
- Normalization: Applied using StandardScaler.
- Split: 70% training and 30% testing.

**Model Architecture and Training (CNN):**
- Conv1D (64 filters) -> BatchNormalization -> MaxPooling1D -> Dropout (0.5)
- Conv1D (128 filters) -> BatchNormalization -> MaxPooling1D -> Dropout (0.5)
- Flatten -> Dense (100 units) -> Dropout (0.5)
- Dense (1 unit, sigmoid activation)
- EarlyStopping: Used with patience of 10.
- Epochs: Up to 3000.
- Batch Size: 512.

**CNN Results:**
- Test Accuracy: 85.06%
- Precision: 0.84
- Recall: 0.8936
- F1 Score: 0.8660
- MCC: 0.6992
- ROC AUC Score: 0.8864
''')

# Add a footer
