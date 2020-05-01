import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('Data.csv')
# print(dataset)
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
'''el 5ana el 2ola fe .iloc[] ma3naha kol el sfof hatha
el 5ana el tanya fe iloc ma3naha hat kol el columns ma3da a5r wa7ed 
y3n3 mfrod el output ykon , .values de bt5le el output fe array
garab etba3 beha mra wmn 8erha mra '''
# print(x)
# print(y)



