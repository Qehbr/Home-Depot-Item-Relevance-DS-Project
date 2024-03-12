# Home-Depot-Item-Relevance-DS-Project
## Overview
This repository is dedicated to the Home Depot Item Relevance project, which aims to determine relevance of the item according to search query. The project focuses on developing models using different NLP techniques: Classical ML, Character Based, Word Based, Pretrained, Combined
## Files
### DoubleLSTM_utils 
*  DoubleLSTMDataset.py - Dataset for DoubleLSTMSiamese Model
*  DoubleLSTMSiameseLSTM.py - DoubleLSTMSiamese Model
### bart_utils
*  BartDataset.py - Dataset for BartSiamese Model
*  BartSiamese.py - BartSiamese Model
*  bart_utils.py - util functions for Bart
### char_utils
*  CharDataset.py - Dataset for CharSiameseLSTM Model
*  CharSiameseLSTM.py - CharSiameseLSTM Model
*  char_utils.py - util functions for character-based model
### csv - contains csv files from Kaggle Competition
### utils
* ClassicalML.py - contains all funciton used for training classical ML algorithms
* GLOBALS.py - contains all global variables
* new_preproc.py - contains new prepocessing functions
* old_preproc.py - contains old prepocessing functions
### word_utils
*  WordDataset.py - Dataset for WordSiameseLSTM Model
*  WordSiameseLSTM.py - WordSiameseLSTM Model
*  word_utils.py - util functions for word-based model
### main
*  2LSTM.ipynb - main for training double LSTM model
*  Bart.ipynb - main for training Bart-based model
*  Character.ipynb - main for training character-based model
*  Naive.ipynb - main for training naive model for comparison
*  Word.ipynb - main for training word-based model

## Dataset
[Link to Kaggle Dataset](https://www.kaggle.com/c/home-depot-product-search-relevance/data)
### Structure
The dataset is composed of different features about items and search queries. In our project we used:
*  Product Descriptions
*  Search Terms
*  Relevance of the search to item description


## Trainining
### Character based model:
* Model Structure:
  
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/d2ce40f2-7b0b-4b78-88ce-2abf3bfa4f20)
*  Train/Validation graphs of RMSE/MAE of best experiment:
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/8c5d8379-f668-43df-b406-b553f4126e17)
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/9314e3a6-33c5-427e-8222-29135f424dfd)
* Results:
  
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/ace4a80f-1b15-48af-9132-fd2ce6bd3ff0)

### Word based model:
* Model Structure:
  
Same as in characted-based model
*  Train/Validation graphs of RMSE/MAE of best experiment:
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/6461e613-6379-47b0-b8d5-efd35baf679f)
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/e8f60782-2646-4f5e-aa05-4c87fd400380)
* Results:
  
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/fccbe15a-9a78-4384-825e-47818f667e85)

### Double LSTM model:
* Model Structure:

Same as in characted-based model but with 2 LSTM based on input
* Results:
  
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/048b77a2-467e-45f1-ab2a-9ae1d7e7df62)

### Classical ML on word-based model:
* Results:
  
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/f7a2f14c-b1f6-4714-b506-413dcf7589b1)

### Bart based model:
* Model Structure:
  
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/4352a4b1-8529-464b-8473-efe43bce5c5d)
*  Train/Validation graphs of RMSE/MAE of best experiment:
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/8e06b0cd-d523-453b-9693-582f5a1e0bcb)
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/1bd33554-d8d2-46da-bac3-f5ad46d7befa)
* Results:
  
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/37e6639d-d57e-4955-b12c-5dbecd56296a)

### Classical ML on Bert-based model:
* Results:
  
  ![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/c56cad82-7bdf-4bdf-8d8c-392b6ba7e045)


## Final Results
![image](https://github.com/Qehbr/Home-Depot-Item-Relevance-DS-Project/assets/49615282/64356e11-78fa-4530-97eb-eb593860fab0)

## Final Remarks
* The project was really challenging, especially preprocessing the data. We think the reason our word-based model got the best results is because of good data preprocessing tuned especially for the task. Although Bert is a very strong and complex model, it trained on very different text and not only Home Depot items, that is the possible reason why it did not outperformed our model. 
It is important to mention that training Bert model was much faster than model from zero, so it is always trade-off between the quality of the model and time.



