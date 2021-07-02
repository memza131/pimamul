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

    st.write(' ---')
    st.title('Model')
    st.write('The model performance of the Pima Diabetes dataset.')
    st.write('You can compare three model using the Pima Diabetes dataset. ')
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

    #choose Parameter 
    def add_parameter_ui(clf_name) :
        params1 = dict()
        if clf_name == 'KNN' :
            K = st.sidebar.slider('K',1,15)
            params1['K'] = K
        elif clf_name == 'SVM' :
            C = st.sidebar.slider('C',0.01,10.0)
            params1['C'] = C 
        else :
            max_depth = st.sidebar.slider('max_depth' , 2 ,15)
            n_estimators = st.sidebar.slider('n_estimators',1,100)
            params1['max_depth' ] = max_depth 
            params1['n_estimators' ]  = n_estimators
        return params1
    params1 = add_parameter_ui(classifier_name )

    #create model
    def get_classifier(clf_name,params) :
        if clf_name == 'KNN' :
            clf = KNeighborsClassifier(n_neighbors = params['K'])
        elif clf_name == 'SVM' :
            clf = SVC(C=params['C'])
        else :
            clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth']
                                        , random_state=1234)
        return clf 

    model = get_classifier(classifier_name,params1)

    #classification  train model
    X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size = 0.2, random_state=1234)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)

    st.write(f'`classifier` = {classifier_name} ')
    st.write(f'`accurancy` = {acc} ')


    
