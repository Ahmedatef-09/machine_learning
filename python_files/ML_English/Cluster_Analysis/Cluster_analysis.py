import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans
sns.set()
counties = ['Usa','Canada','France','Uk','Germany','Australia']
Lattitude = [44.97,62.40,46.75,54.01,51.15,-25.45]
Longitude = [-103.77,-96.8,2.40,-2.53,10.40,133.11]
Language  = ['English','English','French','English','German','English']
data = pd.DataFrame({   'Countries':counties,
                        'Lattitude':Lattitude,
                        'Longitude':Longitude,
                        'Language' :Language })
# print(data)
# plt.scatter(data['Longitude'],data['Lattitude'])
# plt.xlim(-180,180)
# plt.ylim(-90,90)
# plt.show()
'''this is method to save data in x '''
# x = data[['Lattitude','Longitude']]
# print(x)
'''lets try another way '''
x = data.iloc[:,1:3]
# print(x)
'''lets perform clustering '''
kmeans = KMeans(3)
kmeans.fit(x)
cluster_identifier = kmeans.fit_predict(x)
# print(cluster_identifier)
data_with_cluster = data.copy()
data_with_cluster['cluster'] = cluster_identifier
# print(data_with_cluster)
# plt.scatter(data['Longitude'],data['Lattitude'],c=data_with_cluster['cluster'],cmap='rainbow')
# plt.xlim(-180,180)
# plt.ylim(-90,90)
# plt.show()
'''to choose the optimum number of clusters you get wcss'''
wcss = []
for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
'''use the elbow method '''
cluster_numbers = range(1,7)
plt.plot(cluster_numbers,wcss)
plt.show()
'''after show you will see the optimum number is 3'''