# This script is the UI for the project (supports Python 3.8+)
# NOTE: you must clone the Blink-1470 repo (https://github.com/damoon843/Blink-1470) to have display images/audio files
import numpy as np
import pandas as pd
import streamlit as st
import os

st.title("Blink-1470: Sound of Music 🎵🎧")
st.header("*An Interpretable CNN Model for Genre Classification*")

"---"

st.markdown(
    """
    There are **10 genres** that this model seeks to classify:  
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
    **NOTE:** please refer to the poster for more specifics on model architecture, implementation details, etc (found in the repo).
    """
    )

st.subheader("Explore our Results")

# Setup for displaying media
abspath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/')

st.markdown(
	"""
	Feel free to analyze the differences in .wav files and deconvolved activations between the 
	masked and unmasked models. You can also compare the differences in sound and 
	activations between a single filter (out of 32) of a model's first convolution layer.    
	"""
)

genre_type = st.selectbox("Which genre would you like to explore?", 
                          ("Blues", "Jazz", "Classical", "Metal", "Country",
                           "Pop", "Disco", "Reggae", "Hiphop", "Rock"))

masked, unmasked = st.beta_columns(2)
genre_type = genre_type.lower() # Lowercase for templating

with masked:
    st.write("**Masked model**")
    masked_deconv_image = os.path.join(abspath, f'masked/masked_deconv_{genre_type}.png')
    st.image(masked_deconv_image)
    
    masked_deconv_audio = os.path.join(abspath, f'masked/masked_deconv_{genre_type}.wav')
    st.audio(masked_deconv_audio)
        
    st.write("Filter 16 after 1st conv layer:")
    masked_deconv_filter_image = os.path.join(abspath, f'masked/masked_conv1_filter16_{genre_type}.png')
    st.image(masked_deconv_filter_image)
    
    masked_deconv_filter_audio = os.path.join(abspath, f'masked/masked_conv1_filter16_{genre_type}.wav')
    st.audio(masked_deconv_filter_audio)
with unmasked:
    st.write("**Unmasked model**")
    unmasked_deconv_image = os.path.join(abspath, f'unmasked/deconv_{genre_type}.png')
    st.image(masked_deconv_image)
    
    unmasked_deconv_audio = os.path.join(abspath, f'unmasked/deconv_{genre_type}.wav')
    st.audio(unmasked_deconv_audio)
        
    st.write("Filter 16 after 1st conv layer:")
    unmasked_deconv_filter_image = os.path.join(abspath, f'unmasked/conv1_filter16_{genre_type}.png')
    st.image(unmasked_deconv_filter_image)
    
    unmasked_deconv_filter_audio = os.path.join(abspath, f'unmasked/conv1_filter16_{genre_type}.wav')
    st.audio(unmasked_deconv_filter_audio)

st.subheader("References")
st.write(
    """
    The repo for this project can be found [here](https://github.com/damoon843/Blink-1470)  
    The dataset used to train/test this model can be found [here](http://marsyas.info/downloads/datasets.html)  
    The masking layer and general architecture was implemented from [this paper](https://arxiv.org/abs/1710.00935) 
    """
    )