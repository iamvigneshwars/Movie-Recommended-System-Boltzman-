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
