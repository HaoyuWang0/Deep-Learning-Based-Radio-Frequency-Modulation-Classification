# Deep Learning Based Radio Frequency Modulation Classification
ECE 228 Group Project, Spring 2022

## 1. Introduction:
In the domain of Radio Frequency (RF) signal processing, modulation recognition is a task of classifying the modulation of received radio signal. It can help receivers better understand the types of communication scheme and emitter. With the advancement of machine learning techniques, experiments prove that neural network models have promising results in this classification task, which can be treated as a multi-label classification problem. In this project, we implemented multiple deep learning models, including CNN, LSTM, ResNet and CNN-LSTM models, and compare their performance.


## 2. Built-with
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
 | -- CNN_2016.npz
 | -- CNNLSTM_2016.npz
 | -- CNNLSTM_2018.npz
 | -- LSTM_2016.npz
 | -- LSTM_2018.npz
 | -- modulations.pickle
 | -- ResNet33_2018.npz
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
```

Description:

In the 'data' folder, 'Download_Datasets.txt' provides the link to download data. 'data_generator.py' can generate the data subset of RADIOML 2018.10A dataset.

The 'models' folder saves all models' parameters after training.

The 'results' folder saves all training results and the 'Visualizations.ipynb' visualizes all results in a Jupyter Notebook.

The 'source' folder contains all source code for training models.
