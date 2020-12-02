# This script is the UI for the project (supports Python 3.8+)
# To run this file: streamlit run app.py
# Type ctrl+c in the terminal to close the web browser
# NOTE: you must clone the Blink-1470 repo (https://github.com/damoon843/Blink-1470) to have display images/audio files
import numpy as np
import pandas as pd
import streamlit as st
import os
from PIL import Image

st.title("Blink-1470: Sound of Music 🎵🎧")
st.header("*An Interpretable CNN Model for Genre Classification*")

"---"

st.subheader("The Basics...")
st.markdown(
    """
    There are **10 genres** that our model seeks to classify:  
	1. Blues  
	2. Jazz  
	3. Classical  
	4. Metal  
	5. Country  
	6. Pop  
	7. Disco  
	8. Reggae  
	9. Hiphop  
	10. Rock  
	"""
)

st.markdown(
    """
    The general architecture of our model is as follows:  
    *insert model description here*  
    *use keras plot_model to visualize*
    """
    )

st.markdown(
    """
    This model implements a masking layer to filter out noisy activations.
    Here's an example of a single mask, as well as an 
    input feature map's activations before & after its application:    
    *use np colormaps to visualize here*
    """
    )

st.subheader("Interpreting Our Model...")

# Setup for displaying media
abspath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/')

st.markdown(
	"""
	Feel free to explore the differences in .wav files and activations between the 
	masked and unmasked models. You can also compare the differences in sound and 
	activations between a single filter of a model's first convolution layer.    
	*do we want filters after second conv layer?*  
    *do we want to add original audio here?*
	"""
)

genre_type = st.selectbox("Which genre would you like to explore?", 
                          ("Blues", "Jazz", "Classical", "Metal", "Country",
                           "Pop", "Disco", "Reggae", "Hiphop", "Rock"))

if genre_type == "Blues":
    masked, unmasked = st.beta_columns(2)
    with masked:
        st.write("**Masked model**")
        masked_deconv_blues_image = os.path.join(abspath, 'masked/masked_deconv_blues.png')
        try:
            st.image(masked_deconv_blues_image)
        except FileNotFoundError:
            st.write("Error: no image at this path")
        
        masked_deconv_blues_audio = os.path.join(abspath, 'masked/masked_deconv_blues.wav')
        try:
            st.audio(masked_deconv_blues_audio)
        except FileNotFoundError:
            st.write("Error: no audio at this path")
            
        st.write("Filter 16 (out of 32) after 1st conv layer:")
        masked_deconv_blues_filter_image = os.path.join(abspath, 'masked/masked_conv1_filter16_blues.png')
        try:
            st.image(masked_deconv_blues_filter_image)
        except FileNotFoundError:
            st.write("Error: no image at this path")
        
        masked_deconv_blues_filter_audio = os.path.join(abspath, 'masked/masked_conv1_filter16_blues.wav')
        try:
            st.audio(masked_deconv_blues_filter_audio)
        except FileNotFoundError:
            st.write("Error: no audio at this path")
            
    with unmasked:
        st.write("**Unmasked model**")
        unmasked_deconv_blues_image = os.path.join(abspath, 'unmasked/deconv_blues.png')
        try:
            st.image(unmasked_deconv_blues_image)
        except FileNotFoundError:
            st.write("Error: no image at this path")
        
        unmasked_deconv_blues_audio = os.path.join(abspath, 'unmasked/deconv_blues.wav')
        try:
            st.audio(unmasked_deconv_blues_audio)
        except FileNotFoundError:
            st.write("Error: no audio at this path")
        
        st.write("Filter 16 (out of 32) after 1st conv layer:")
        masked_deconv_blues_filter = os.path.join(abspath, 'unmasked/conv1_filter16_blues.png')
        try:
            st.image(masked_deconv_blues_filter)
        except FileNotFoundError:
            st.write("Error: no image at this path")
        
        unmasked_deconv_blues_filter_audio = os.path.join(abspath, 'unmasked/conv1_filter16_blues.wav')
        try:
            st.audio(unmasked_deconv_blues_filter_audio)
        except FileNotFoundError:
            st.write("Error: no audio at this path")

# TODO: insert media for all other genres
elif genre_type == "Jazz":
    pass
elif genre_type == "Classical":
    pass
elif genre_type == "Metal":
    pass
elif genre_type == "Country":
    pass
elif genre_type == "Pop":
    pass
elif genre_type == "Disco":
    pass
elif genre_type == "Reggae":
    pass
elif genre_type == "Hiphop":
    pass
elif genre_type == "Rock":
    pass

st.subheader("References...")
st.write(
    """
    The repo for this project can be found [here](https://github.com/damoon843/Blink-1470)  
    The dataset used to train/test this model can be found [here](http://marsyas.info/downloads/datasets.html)  
    The masking layer and general architecture was implemented from [this paper](https://arxiv.org/abs/1710.00935) 
    """
    )