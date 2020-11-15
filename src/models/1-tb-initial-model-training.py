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
#from jupytext.config import find_jupytext_configuration_file
#find_jupytext_configuration_file('.')

# %% [markdown]
# # Entraînement du modèle

# %%
# Import pour les tests unitaires
import pytest

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
                               n_jobs=1, 
                               solver='liblinear')

# Train the Model
LR_Model.fit(Xtrain, Ytrain)  

# %%
# Calculate the Score 
score = LR_Model.score(Xtest, Ytest)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = LR_Model.predict(Xtest)  

print(Ypredict)
