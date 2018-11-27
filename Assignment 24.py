
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target


# In[2]:


print(features.shape)
features.head()


# In[20]:


# Now we have to standardize the data. We will use StandardScaler from preprocessing

X_train,X_test,y_train,y_test=train_test_split(features,targets,test_size=.25,random_state=18)

scaler= StandardScaler().fit(X_train)

X_train_scaled=pd.DataFrame(scaler.transform(X_train),index=X_train.index.values,columns=X_train.columns.values)

X_test_scaled=pd.DataFrame(scaler.transform(X_test),index=X_test.index.values,columns=X_test.columns.values)


# In[16]:


# we can even check if we can reduce dimensionality of dataset using PCA, although here its only 13 features.
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
pca=PCA()
pca.fit(X_train)
foo=pd.DataFrame(pca.transform(X_train))
x_axis=np.arange(1,pca.n_components_ + 1)
pca_scaled=PCA()
pca_scaled.fit(X_train_scaled)
foo_scaled=pd.DataFrame(pca.transform(X_train_scaled))
foo_scaled.hist()
plt.show()


# In[18]:


# import ,instantiate,fit
from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=1000,oob_score=True,random_state=18)
# oob_True :whether to use out-of-bag samples to estimate
#    the R^2 on unseen data.is kind of cross-validation
forest.fit(X_train,y_train)


# In[19]:


from sklearn.metrics import r2_score
preds=forest.predict(X_test)
test_score=r2_score(y_test,preds)
print(' R squared score:',test_score)

