# Blink-1470
CSCI 1470 Final Project

Please refer to the poster (found in this repo) for project details, implementation, and results.

**UI usage:**  
To visualize and compare activations between the base CNN model and the masked CNN model, please reference the UI (built on streamlit). It is important to note that to properly play the .wav files and display the activatiosn, this repo must be cloned prior to running. Please make sure to activate the virtual environment (with Python 3.8) and run the requirements script with the following commands:
```
source .venv/bin/activate
pip install -r requirements.txt
```
To run the UI (in your localhost), you can run the following command from the scripts folder: 
```
streamlit run app.py
```
Please refer to the [Streamlit docs](https://docs.streamlit.io/en/stable/) for more information.
