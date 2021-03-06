# Deep Learning Based Radio Frequency Modulation Classification
ECE 228 Group Project, Spring 2022

## 1. Introduction:
In the domain of Radio Frequency (RF) signal processing, modulation recognition is a task of classifying the modulation of received radio signal. It can help receivers better understand the types of communication scheme and the emitters. With the advancement of machine learning techniques, experiments prove that neural network models have promising results in this classification task, which can be treated as a multi-label classification problem. In this project, we implemented multiple deep learning models, including CNN, LSTM, ResNet and CNN-LSTM models, and compare their performance.


## 2. Built-with
### Install Requirements:
```
$ pip install -r requirements.txt
```
  1. h5py==3.6.0
  2. keras==2.8.0
  3. matplotlib==3.5.0
  4. mlxtend==0.19.0
  5. numpy==1.20.3
  6. scikit_learn==1.1.1
  7. tensorflow==2.8.0
  8. pickle

## 3. File Structure

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
 |
-- requirements.txt
```

### Description:

In the *data* folder, *Download_Datasets.txt* provides the link to download data. The file *data_generator.py* can generate the data subset of *RADIOML 2018.10A* dataset.

The *models* folder saves all models' parameters after training.

The *results* folder saves all training results and *Visualizations.ipynb* visualizes all results in a Jupyter Notebook.

The *source* folder contains all the source code for training models.

## 4. How to Run Our Code
1. Before running every script, make sure the data folder is correct.

2. Download the *RADIOML 2016.10A* and *RADIOML 2018.10A* [datasets](https://www.deepsig.ai/datasets), and put them in data folder.

3. Preprocess the 2018 dataset with *data_generator.py*.
```
$ python data/data_generator.py
```

4. Run all source code with
```
$ python source/${file_name}.py
```

5. To visualize all training results, open a Jupyter Notebook with
```
$ jupyter notebook results/Visualizations.ipynb
```
and run all the cells in *Visualizations.ipynb*
