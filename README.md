# Kaggle-Automated-Essay-Checking-System  
My attempt for the Kaggle AES project (https://www.kaggle.com/c/asap-aes)
The linear regression model uses a Word2Vec model and custom generated heuristic features to obtain a mean-quadratic-weighted-kappa score of 0.9359.

### Notebooks:  
1. CustomFeatureGeneration.ipynb - Generating custom features for the data set.
2. Data_Exploration.ipynb - Exploring the data set and free form visualization.
3. Linear Regression Model.ipynb - The model building and learning takes place here.  

### Helper functions and required library import:  
./utils/helperfunctions.py  
./utils/requirements.py  

### Libraries used for the Capstone Project:  
1. Scikit-learn 0.18.1: pip install --user --upgrade sklearn
2. Gensim 2.1.0: pip install --user --upgrade gensim
3. Textmining 1.0: pip install --user --upgrade textmining
4. Grammar Check 1.3.1 : 1. pip install --upgrade 3to2
				2. pip install --user --upgrade grammar-check
5. Matplotlib 2.0.0: pip install --user --upgrade matplotlib
6. NLTK 3.2.2: pip install --user --upgrade nltk

**NOTE** You need to download all the NLTK's data first inorder to use its packages, to do so type following commands in python (referece: http://www.nltk.org/data.html)
  - ```import nltk```
  - ```nltk.download()```
- You also need Java installed on your machine to run NLTK.
  Java installation steps for Ubuntu 16.04 : (http://www.wikihow.com/Install-Oracle-Java-on-Ubuntu-Linux)

7.** Dataset : domain123.csv  **
- ** Images and saved models : ```./model_and_visualization/```**   
- ** References : ```./References/ **```  
- ** Essay set description : ```./Essay_Set_Descriptions``` **