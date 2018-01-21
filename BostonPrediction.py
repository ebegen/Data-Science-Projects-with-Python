
# coding: utf-8

# In[56]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().magic('matplotlib inline')


# In[9]:


dataset = load_boston()
print(dataset.DESCR)


# In[19]:


boston_df = pd.DataFrame(data=dataset.data, columns= dataset.feature_names)
boston_df['price'] = dataset.target 
boston_df.info()


# In[20]:


boston_df.describe()


# In[22]:


sns.set_context("poster")
sns.pairplot(boston_df)


# In[24]:


sns.distplot(boston_df['price'])


# In[26]:


sns.heatmap(boston_df.corr())


# In[29]:


X= boston_df.drop('price',axis=1)
y= boston_df['price']


# In[34]:


X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.33,random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[37]:


lm = LinearRegression()
lm.fit(X_train,Y_train)


# In[38]:


Y_pred = lm.predict(X_test)


# In[42]:


plt.scatter(Y_test, Y_pred, s=5)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")


# In[53]:


# print the intercept
print(lm.intercept_)


# In[54]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[60]:


sns.distplot((Y_test-Y_pred),bins=50);


# In[57]:


print('MAE:', metrics.mean_absolute_error(Y_test, Y_pred))
print('MSE:', metrics.mean_squared_error(Y_test, Y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

