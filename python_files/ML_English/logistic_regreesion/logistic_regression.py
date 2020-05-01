import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


raw_data = pd.read_csv('2.02. Binary predictors.csv')
# print(raw_data)
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1,'No':0})
data['Gender'] = data['Gender'].map({'Female':1,'Male':0})
# print(data)
y = data['Admitted']
x1 = data[['SAT','Gender']]
# print(x1)
# '''lets create the regression using stats model'''
x = sm.add_constant(x1)
# # print(x)
reg_log = sm.Logit(y,x) #this is logistic_regression code line
result_log = reg_log.fit()
# # print(reg_log)
# print(reg_log.summary())
'''if we take np.exp(gender_coef) it will result 7 
which mean in the same sat score female has 7 times higher odd than males'''
#if we want to predict values the use reg_log.predict()
np.set_printoptions(formatter={'float':lambda x: "{0:0.2f}".format(x)}) #format output
# print(result_log.predict()) 
'''if you want to compare result predicted with actual you use this method'''
# print(result_log.pred_table())
'''lets try to show the table in gopod look '''
df = pd.DataFrame(result_log.pred_table(),columns=['predicted 0 ','predicted 1'],index=['actual 0 ','actual 1'])
# print(df)
'''to calculate the accuracy '''
accuracy_df = np.array(df)
accuracy_final = (accuracy_df[0,0]+accuracy_df[1,1])/accuracy_df.sum()
# print(accuracy_final)
'''now lets test our model '''
# print(x)
