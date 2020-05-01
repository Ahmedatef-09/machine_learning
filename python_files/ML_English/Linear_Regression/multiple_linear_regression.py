import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('7.2 1.02. Multiple linear regression.csv.csv')
x = data[['SAT','Rand 1,2,3']]
y = data['GPA']
'''we dont need to reshape x because its now 2-d array '''
reg = LinearRegression() # this is the regreesyion itself
reg.fit(x,y) #fit linear regreesion on x, y 
# as usual you must fit after creating object
'''HOW TO GET RSQUARE '''
# reg.score(x_matrix,y)

'''HOW TO GET Adjusted r2 '''
def adjusted_r():
    r1 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    r2 = 1-(1-r1)*(n-1)/(n-p-1)
    return r2
# print(adjusted_r())
'''now we need to find p values in order to determine whereas independent
variable is significant or not so we use sklearn.feature selection
de ba2a feha function esmha f_regression(x,y) el function de bt3ml 
simple linear regreesion (y,x1),(y,x2)wettala3 f statistic wel p values 
btoo3 kol independant variable '''
p_values = f_regression(x,y)[1]
p_values = p_values.round(3)
# print(p_values)
'''now lets create a summary table '''
reg_summary = pd.DataFrame(x.columns.values,columns=['features'])
reg_summary['coefficient'] = reg.coef_
reg_summary['p-values'] = p_values
# print(reg_summary)
'''each variable x  should be standarized '''
scaler = StandardScaler() # create an object of standardscaler class
scaler.fit(x)             # as usual you must fit after creating object
x_scaled = scaler.transform(x)
# print(x_scaled)
'''now lets do the regression again with standarization '''
reg.fit(x_scaled,y)
reg_summary2 = pd.DataFrame([['INTERCEPT'],['SAT'],['Rand 1,2,3']],columns=['features'])
reg_summary2['values'] = reg.intercept_,reg.coef_[0],reg.coef_[1]
print(reg_summary2)
z = pd.DataFrame([[1700,2],[1800,1]],columns=['SAT','Rand 1,2,3'])

z_scaled = scaler.transform(z)

# print(reg.predict(z_scaled))
'''now we will know test-split '''
a = np.arange(1,101)
b = np.arange(501,601)

a_train,a_test,b_train,b_test = train_test_split(a,b,test_size = 0.2,shuffle = True,random_state = 42)
print(b_train)