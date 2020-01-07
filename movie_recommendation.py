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

# Getting the maximum number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into array with users in low and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:,0] == id_users]
        id_ratings = data[:, 2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data in to Torch Tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
