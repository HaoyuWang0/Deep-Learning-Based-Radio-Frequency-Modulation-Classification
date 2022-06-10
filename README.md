# Deep Learning Based Radio Frequency Modulation Classification
ECE 228 Group Project, Spring 2022

## 1. Introduction:
In the domain of Radio Frequency (RF) signal processing, modulation recognition
is a task of classifying the modulation of received radio signal. It can help
receivers better understand the types of communication scheme and emitter.
With the advancement of machine learning techniques, experiments prove that
neural network models have promising results in this classification task,
which can be treated as a multi-label classification problem. In this project,
we implemented multiple deep learning models, including CNN, LSTM, ResNet and
CNN-LSTM models, and compare their performance.


## 2. Build-with
```
$ pip install -r requirements.txt
```
## 3. File Structures

```
-- data
 | -- data_generator.py
 | -- Download_Datasets.txt
 |
-- models
 | -- models.md
 |
-- results
 | -- Visualizations.ipynb
 |
-- source
 | -- CNN_2016.py
 | -- CNNLSTM_2016.py
 | -- CNNLSTM_2018.py
 | -- LSTM_2016.py
 | -- LSTM_2018.py
 | -- ResNet33_2018.py
 |
-- README.md
-- requirements.txt
```
