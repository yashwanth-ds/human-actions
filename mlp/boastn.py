import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
boston=load_boston()
print(boston.data)
a=boston.feature_names
features=pd.DataFrame(boston.data,columns=a)
target=pd.DataFrame(boston.target,columns=['TARGET'])

data=pd.concat([features,target],axis=1)

data2=data.corr('pearson')
print(data2)
target_f=abs(data2['TARGET']).sort_values(ascending=False)
print(target_f)
