import numpy as np 
import pandas as pd 

dic  =  {'a':[1,2,3],
        'b':[4,5,6],
        'c':[7,8,9],
        'key':'a b c'.split()}


dic2 =  {'a':[10,20,30],
        'b':[40,50,60],
        'c':[70,80,90],
        'key':'a b c'.split()}
df = pd.DataFrame(dic,index=[0,1,2])
df2 = pd.DataFrame(dic2,index=[0,1,2])

df3 = pd.concat([df,df2],axis=0) #paste dataframe,df2 under each other 

df4 = pd.merge(df,df2,how= 'inner',on='key')#inner join df,df2 u can merge more than one ke
#df.join(df2)( in this case merge by index index df = index df2 ) but in .merge merge done by column 
print(df)
