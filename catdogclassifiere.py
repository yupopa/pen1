import streamlit as st
import numpy as np
from PIL import Image
from fastai.vision.all import load_learner, Path
st.title("WELCOME CAT DOG CLASSIFIER")
st.title("cat dog classifer")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
#file upload












if uploaded_file is not None:
    #image transformation and prediciton
    img = Image.open(uploaded_file)
    st.image(img, caption='Your Image.', use_column_width=True)
    image = np.asarray(img)
    label = learn_inf.predict(image) 
    #label[0] accesses the actual label string
    #output display
    st.write("")
    st.write("Classifying...")
    #check for vowels in the names for correct grammar
    if label[0][0] in "AEIOU":
        st.write("## This looks like an")
    else:
        st.write("## This looks like a")
    #our labels are in capital letters only
    st.title(label[0].lower().title())
