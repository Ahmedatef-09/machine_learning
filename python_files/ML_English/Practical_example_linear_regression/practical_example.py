import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('1.04. Real-life example.csv')
# print(data.describe(include = 'all'))
'''the first observation that count is not equal in all of them 
second column models has 312 unique dummy variable which is so many '''
#to get rid of models 
data = data.drop('Model',axis = 1)
# print(data.describe(include = 'all'))
'''now lets check for null values'''
# print(data.isnull().sum())
data_no_mv = data.dropna(axis = 0) #remove all null values in all rows
# print(data_no_mv.describe(include = 'all'))
'''try yo plot each variable in describe to see its disribution '''
'''get rid of outliers of all variables'''
data_2 = data_no_mv[data_no_mv['Price']<(data_no_mv['Price'].quantile(0.99))]
# print(data_2.describe(include = 'all'))

data_3 = data_2[data_2['Mileage']<(data_2['Mileage'].quantile(0.99))]
# print(data_3.describe(include = 'all'))

data_4 = data_3[data_3['EngineV']<6.5]
# print(data_4.describe(include = 'all'))

data_5 = data_4[data_4['Year']>(data_4['Year'].quantile(0.01))]
# print(data_5.describe(include = 'all'))
'''check linearity after ploting we found non linearity we fix it with log'''
log_price = np.log(data_5['Price'])
data_5['log_price'] = log_price
data_5 = data_5.drop('Price',axis = 1)
# print(data_5)
'''--------------------------------------------------------------'''
'''check for multocolinearity using vif factor for variables 
we found that year was too correlated with the other variable so we need to drop it'''
data_no_collinearity = data_5.drop('Year',axis = 1)
# print(data_no_collinearity)
'''check for dummy variables '''
data_with_dimmies = pd.get_dummies(data_no_collinearity,drop_first = True)
# print(data_with_dimmies)
# col = data_with_dimmies.columns.values
col = ['log_price','Mileage','EngineV','Brand_BMW','Brand_Mercedes-Benz',
 'Brand_Mitsubishi','Brand_Renault','Brand_Toyota','Brand_Volkswagen',
 'Body_hatch','Body_other','Body_sedan','Body_vagon','Body_van',
 'Engine Type_Gas','Engine Type_Other','Engine Type_Petrol',
 'Registration_yes']
# print(col)
data_preprocessed = data_with_dimmies[col]
# print(data_preprocessed)
'''lets apply linear regreesion first create x and y '''
y = data_preprocessed['log_price']
x = data_preprocessed.drop('log_price',axis = 1)
# print(x)
'''you need to standarized the inputs '''
scaler = StandardScaler()
scaler.fit(x)
scaled_inputs = scaler.transform(x)
'''split the data to train and test '''
x_train,x_test,y_train,y_test = train_test_split(scaled_inputs,y,test_size = 0.2,random_state = 365)

'''now create linear regreesion model '''
reg = LinearRegression()
reg.fit(x_train,y_train)
'''now it is called logged linear regrresion because the independant variable 
is log price '''
yhat = reg.predict(x_train)
