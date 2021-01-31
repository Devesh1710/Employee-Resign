from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('Final_model')



def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions


def run():
    from PIL import Image
    image = Image.open('C:/Users/Lenovo/Downloads/Case-Study-22-Model-Deployment (1)/Case Study/employee.jpeg')
    image_office = Image.open('C:/Users/Lenovo/Downloads/Case-Study-22-Model-Deployment (1)/Case Study/office.jpeg')
    st.image(image,use_column_width=False)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Realtime", "Batch"))
    st.sidebar.info('This app predicts if an employee will leave the company')
    st.sidebar.image(image_office)
    st.title("Predicting employee leaving")
    if add_selectbox == 'Realtime':
        satisfaction_level=st.number_input('satisfaction_level' , min_value=0.1, max_value=1.0, value=0.1)
        last_evaluation =st.number_input('last_evaluation',min_value=0.1, max_value=1.0, value=0.1)
        number_project = st.number_input('number_project', min_value=0, max_value=50, value=5)
        average_montly_hours=st.number_input('average_montly_hours' , min_value=0.1, max_value=1.0, value=0.1)
        time_spend_company = st.number_input('time_spend_company', min_value=1, max_value=10, value=3)
        Work_accident = st.number_input('Work_accident',  min_value=0, max_value=50, value=0)
        promotion_last_5years = st.number_input('promotion_last_5years',  min_value=0, max_value=50, value=0)
        department = st.selectbox('department' ,['sales', 'accounting', 'hr', 'technical', 'support', 'management','IT', 'product_mng', 'marketing', 'RandD'])
        salary = st.selectbox('salary', ['low', 'high','medium'])
        output=""
        input_dict = {'satisfaction_level':satisfaction_level,'last_evaluation':last_evaluation,'number_project':number_project,'average_montly_hours':average_montly_hours,'time_spend_company':time_spend_company,'Work_accident': Work_accident,'promotion_last_5years':promotion_last_5years,'department':department,'salary' : salary}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

    
    
    
run()