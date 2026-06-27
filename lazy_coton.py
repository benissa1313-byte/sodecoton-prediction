# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:45:54 2026

@author: THIS PC
"""

# Installation si nécessaire
!pip install lazypredict pandas scikit-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lazypredict.Supervised import LazyRegressor
import warnings
warnings.filterwarnings("ignore")
