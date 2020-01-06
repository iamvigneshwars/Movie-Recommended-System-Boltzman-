#!/usr/bin/env python3
# -*- coding: latin-1
"""
Created on Mon Jan  6 20:05:14 2020

@author: iamvigneshwars
"""

# importing the libraries
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
import torch.nn.parallel
import torch.optim.adam
import torch.utils.data
import torch.autograd.variable

# Importing the dataset
movies = pd.read_csv("ml-1m/movies.dat", sep ='::', header = None, engine = "python", encoding = "latin-1")
users = pd.read_csv("ml-1m/users.dat", sep ='::', header = None, engine = "python", encoding = "latin-1")
rating = pd.read_csv("ml-1m/ratings.dat", sep ='::', header = None, engine = "python", encoding = "latin-1")

# preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = "\t")
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = "\t")
test_set = np.array(test_set, dtype = 'int')
