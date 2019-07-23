import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
wine = load_wine()
X= pd.DataFrame(wine.data, columns=wine.feature_names)
 
data1=pd.DataFrame(X['alcohol'])#load the values in alcohol column to data1
data2=pd.DataFrame(X['malic_acid'])#load the values in malic_acid column to data2

data=pd.concat([data1,data2],axis=1)
data.head(5)
plt.scatter(data1,data2,color='green')
#plt.show()
data1_std = np.array(data1)
feature1 = (data1_std - data1_std.mean())/data1_std.std()

data2_std = np.array(data2)
feature2 = (data2_std - data2_std.mean())/data2_std.std()

print(f"Mean after standardization:\nAlcohol={feature1.mean()}, Malic acid={feature2.mean()}")

print(f"standard deviation after standardization:\nAlcohol={feature1.std()}, Malic acid={feature2.std()}")
plt.scatter(feature1,feature2,c='r')
from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit([data1,data2])
df_standardize = std_scale.transform([data1,data2])
data1_norm = np.array(data1)
ft1 = (data1_norm - data1_norm.mean()) / (max(data1_norm)-min(data1_norm))

data2_norm = np.array(data2)
ft2 = (data2_norm - data2_norm.mean()) / (max(data2_norm)-min(data2_norm))

print(f"Mean after standardization:\\nAlcohol={ft1.mean()}, Malic acid={ft2.mean()}")

print(f"Standard deviation after standardization:\\nAlcohol={ft1.std()}, Malic acid={ft2.std()}")
