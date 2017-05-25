import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import re
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import grammar_check
from collections import Counter
import textmining
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from metrics import kappa
import logging
from gensim.models import word2vec

