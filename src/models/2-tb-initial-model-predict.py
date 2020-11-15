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
# # Approach 1 : Pickle approach

# %%
# Import pickle Package

import pickle
import os

# %% [markdown]
# ## Reload du modèle entraîné pour utilisation 

# %%
# Load the Model back from file

filepath = "../models/"

Pkl_Filename = "Pickle_RL_Model.pkl" 

with open(os.path.join(filepath, Pkl_Filename), 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

Pickled_LR_Model

# %%
# Use the Reloaded Model to 
# Calculate the accuracy score and predict target values

# Calculate the Score 
score = Pickled_LR_Model.score(Xtest, Ytest)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_LR_Model.predict(Xtest)  

Ypredict

# %% [markdown]
# # Approach 2 - Joblib

# %%
# Import Joblib Module from Scikit Learn

from sklearn.externals import joblib

# %% [markdown]
# ## Reload du modèle entraîné pour utilisation 

# %%
# Load from file

filepath = "../models/"
joblib_file = "joblib_RL_Model.pkl" 
file = os.path.join(filepath, joblib_file)

joblib_LR_model = joblib.load(file)


joblib_LR_model

# %%
# Use the Reloaded Joblib Model to 
# Calculate the accuracy score and predict target values

# Calculate the Score 
score = joblib_LR_model.score(Xtest, Ytest)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = joblib_LR_model.predict(Xtest)  

Ypredict
