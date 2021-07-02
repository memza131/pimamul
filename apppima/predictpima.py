import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import pandas as pd

def app():

    
    # Load pima dataset
    def load_data() :
            data = pd.read_csv('pima-data.csv')
            X = data.drop(columns=['diabetes']).values
            y = data['diabetes'].values
            return X,y

    #preprocessing data
    X,y = load_data() 

    #ทำsidebox เลือก parameter 
    classifier_name = st.sidebar.selectbox('Select Classifier' , ('KNN' , 'SVM' , 'Random Forest') )

    #create model
    def get_classifier(clf_name) :
        if clf_name == 'KNN' :
            clf = KNeighborsClassifier(n_neighbors=6)
        elif clf_name == 'SVM' :
            clf = SVC(C=4)
        else :
            clf = RandomForestClassifier(n_estimators=6, max_depth=6 , random_state=1234)
        return clf 

    model = get_classifier(classifier_name)

    
    #note
    st.write(' ---')
    st.write(' # Prediction')
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
    
    

    #Block of your data 
    st.sidebar.header('what is your Data')
    def user_feature() :
        pregnant = st.sidebar.slider('Pregnant',0,15,2)
        glucose  = st.sidebar.slider('Glucose',0,200,50)
        blood    = st.sidebar.slider('BloodPressure',0,200,50)
        skin     = st.sidebar.slider('SkinThickness',0,100,20)
        insulin  = st.sidebar.slider('Insulin',0,800,10)
        bmi      = st.sidebar.slider('BMI',0.0,70.0,10.0)
        dpf      = st.sidebar.slider('DPF',0.000,2.000,0.005)
        age      = st.sidebar.slider('Age',0,100,10)

        data = {'Pregnant' : pregnant , 
                'Glucose' : glucose ,
                'BloodPressure' : blood , 
                'SkinThickness' : skin ,
                'Insulin' : insulin,
                'BMI' : bmi,
                'DPF' : dpf,
                'Age' : age}
        features = pd.DataFrame(data,index=[0])
        return features


    features = user_feature()

    st.write('Specified Your Data')
    st.write(features)
    st.write('-----')

    
    #classification  train model

    X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size = 0.2, random_state=1234)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)

    #predict data ที่รับมา
    predict = model.predict(features)

    st.write('**Prediction**')
    st.write(f'`classifier` = {classifier_name} ')
    st.write(f'`accuracy` = {acc} ')

    output = ''
    if predict[0] == 0 :
        output = 'Hurray! You do not have diabetes.'
    else :
        output = 'Sorry ! You have diabetes.'

    st.write('**`Your result`** :' , output)

    
    