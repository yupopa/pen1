from fastai.vision.widgets import *
from fastai.vision.all import *
from pathlib import Path
import streamlit as st
from urllib.request import urlretrieve
url = 'https://www.dropbox.com/s/6j8y83elzpenz1c/export.pkl?raw=1'
filename = 'export.pkl'
urlretrieve(url,filename)
st.markdown("HIGHLIGHTER OR PENCIL")

class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()
            
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')

    def get_prediction(self):

        if st.button('Classify'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
        else: 
            st.write(f'Click the button to classify') 

if __name__=='__main__':
    predictor = Predict(filename)
    
