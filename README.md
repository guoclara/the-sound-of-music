# Blink-1470
CSCI 1470 Final Project

References:
The main reference for this project is the public implementation of the interpretable CNN paper we are following: https://github.com/andrehuang/InterpretableCNN
As discussed with our TA, we will only referencing it in line with Final Project Guidelines:
a) We are using Tensorflow 2 and Keras rather than Tensorflow 1.
b) we are applying the model to a different dataset (spectrograms rather than visual images)

Another source of possible references for our project are these Kaggle notebooks: https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification/notebooks. We do not plan on referencing these for our project, but have listed them here for transparency, since our dataset is somewhat commmon.

**UI usage:**  
To visualize and compare activations between the base CNN model and the masked CNN model, please reference the UI (built on streamlit). Please make sure to activate the virtual environment and run the requirements script with the following commands:
```
source .venv/bin/activate
pip install -r requirements.txt
```
To run the UI (in your localhost), you can run the following command from the scripts folder: 
```
streamlit run app.py
```
Please refer to the [Streamlit docs](https://docs.streamlit.io/en/stable/) for more information.
