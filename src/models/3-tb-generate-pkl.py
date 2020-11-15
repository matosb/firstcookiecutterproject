# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### From https://www.kaggle.com/prmohanty/python-how-to-save-and-load-ml-models

# %%
from jupytext.config import find_jupytext_configuration_file
find_jupytext_configuration_file('.')

# %% [markdown]
# # Entraînement du modèle

# %%
# Import Required packages 
#-------------------------

# Import the Logistic Regression Module from Scikit Learn
from sklearn.linear_model import LogisticRegression  

# Import the IRIS Dataset to be used in this Kernel
from sklearn.datasets import load_iris  

# Load the Module to split the Dataset into Train & Test 
from sklearn.model_selection import train_test_split

# %%
# Load the data
Iris_data = load_iris()

# %%
# Split data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Iris_data.data, 
                                                Iris_data.target, 
                                                test_size=0.3, 
                                                random_state=4)  

# %%
# Define the Model
LR_Model = LogisticRegression(C=0.1,  
                               max_iter=20, 
                               fit_intercept=True, 
                               n_jobs=3, 
                               solver='liblinear')

# Train the Model
LR_Model.fit(Xtrain, Ytrain)  

# %% [markdown]
# # Approach 1 : Pickle approach

# %% [markdown]
# ## Sauvegarde du modèle entraîné

# %%
# Import pickle Package

import pickle

# %%
# Save the Modle to file in the current working directory
import os

filepath = "../models/"

Pkl_Filename = "Pickle_RL_Model.pkl"  

with open(os.path.join(filepath, Pkl_Filename), 'wb') as file:  
    pickle.dump(LR_Model, file)

# %% [markdown]
# # Approach 2 - Joblib

# %% [markdown]
# ## Sauvegarde du modèle entraîné

# %%
# Import Joblib Module from Scikit Learn

from sklearn.externals import joblib

# %%
# Save RL_Model to file in the current working directory

filepath = "../models/"
joblib_file = "joblib_RL_Model.pkl"  

file = os.path.join(filepath, joblib_file)

joblib.dump(LR_Model, file)
