import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression


data = pd.read_csv('4.2 1.01. Simple linear regression.csv.csv')
x = data['SAT']
y = data['GPA']
'''sklearn doesnt deal with 1d array so we need to reshape it to 2-d array'''
x_matrix = x.values.reshape(-1,1)
reg = LinearRegression() # this is the regreesyion itself
reg.fit(x_matrix,y) #fit linear regreesion on x, y 
# print(x_matrix.shape)
'''HOW TO GET RSQUARE '''
# reg.score(x_matrix,y)

'''THE COEFFIOSONTS B1,B2,.....'''
# reg.coef_ # output is 1-d array filled with the coefficients

''' to get the intercept b0 '''
# reg.intercept_

'''to make predictions based on your regression '''

print(reg.predict(np.array([1740]).reshape(-1,1)))#reg.pretect must take 2-d array input
'''lets give an example'''
# new_data_frame = pd.DataFrame([1740,1760],columns=['SAT'])
# new_data_frame['pregicted gpa'] = reg.predict(new_data_frame)
# print(new_data_frame)
'''Now lets start to plot the data '''
plt.scatter(x_matrix,y)
yhat = reg.intercept_+reg.coef_*x_matrix
plt.plot(x,yhat,lw=4,c='red',label= 'regression ')
plt.ylabel('GPA',fontsize = 20)
plt.ylabel('SAT',fontsize = 20)
plt.show()