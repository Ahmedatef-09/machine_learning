import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 


df = pd.read_csv('4.2 1.01. Simple linear regression.csv.csv')

#print(df.describe()) # describe more information about df such as mean mod etc...
#you can also read from excel pd.read_excel('name',sheet = 'sheet_name')
#you can convert between files df.excel('new excell file_name',sheet_name = 'sheet_name')

df2 = pd.read_html('https://www.w3schools.com/html/html_tables.asp')
'''when you use pd.read for html pandas search in the url untill it find table and 
put that file in df2 
'''

'''lets start to build our first regression dont forget y = b0+b1x1
in our case y = gpa and x is the observet SAT'''

y = df['GPA']
x1 = df['SAT']
#the next commants is to plot the data 
plt.scatter(x1,y) # x- axis and y-axis
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
# plt.show()
#to perform linear regression we use statsmodel library 
#note : you can use any lib to perform linear regression such as numpy statsmodel etc>>
#to create linear regreesion using stats model we type the following 
x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
print(result.summary()) #we found b0 and b1 from result
yhat = 0.0017*x1+0.0275
fig = plt.plot(x1,yhat,lw=4,c='orange',label = 'regression line')
plt.xlim(0)
plt.ylim(0)
plt.show()