#!/usr/bin/env python
# coding: utf-8

# In[62]:


#kutuphaneler importlandi
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score




import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams


# In[30]:


from sklearn.datasets import load_boston #data setimi importladim
boston = load_boston()


# In[31]:


print(boston.keys()) 


# In[32]:


print(boston.data.shape) #matrix kaca kaclik oldugunu gordum 


# In[33]:


print(boston.feature_names)


# In[34]:


print(boston.DESCR)#aciklamalar


# In[51]:


bos = pd.DataFrame(boston.data)
print(bos.head()) #head komutu listedeki en ust 20 parametreyi cekiyor


# In[50]:


bos.columns = boston.feature_names
print(bos.head())


# In[37]:


bos['PRICE'] = boston.target
print(bos.head())


# In[53]:


print(bos.describe())


# In[39]:


X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']


# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=101)


# In[63]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Fiyat: $Y_i$")
plt.ylabel("Tahminlenen Fiyat: $\hat{Y}_i$")
plt.title("Fiyat vs Tahminlenen fiyat: $Y_i$ vs $\hat{Y}_i$")


# In[47]:


mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)


# In[64]:



# Predicition doğruluk oranım
print('Variance score: %.2f' % r2_score(Y_test, Y_pred))


# In[ ]:




