import streamlit as st
from multiapp import MultiApp
from apppima import homepima, datapima, modelpima , predictpima # import your app modules here

apppy = MultiApp()

st.write(""" # Hello """)

# Add all your application here
apppy.add_app("Home", homepima.app)
apppy.add_app("Data", datapima.app)
apppy.add_app("Test Model", modelpima.app)  
apppy.add_app("Prediction", predictpima.app)
# The main app


apppy.run()