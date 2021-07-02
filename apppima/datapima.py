import streamlit as st
import numpy as np
import pandas as pd


def app():
    st.write(' ---')     
    
    st.title('Data')

    st.write(" ** This is the DataFrame of the `Pima Diabetes` dataset. ** ")

    st.write('Number of observations  : 768')
    st.write('Number of columns : 9')

    #note
    st.write(' ---')
    st.write(' **Note**')
    st.write('`Pregnant`       : Number of times pregnant')
    st.write('`Glucose`        : Plasma glucose concentration (2hour)')
    st.write('`BloodPressure`  : Diastolic blood pressure (mm Hg)')
    st.write('`SkinThickness`  : Triceps skin fold thickness (mm)')
    st.write('`Insulin`        : 2-Hour serum insulin (mu U/ml)')
    st.write('`BMI`            : Body mass index')
    st.write('`DPF`            : Diabetes pedigree function ')
    st.write('`Age`            : Age (year)')
    
    st.write('---')

    #load data
    def load_data() :
            data = pd.read_csv('pima-data.csv')
            return data
    data_pima = load_data() 
    num = st.slider('Example(1-100) : Pima Diabete Data',0,100,5)
    st.write(data_pima[:num])
    

   
