#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import seaborn as sns
sns.set(style="whitegrid")


# In[2]:


import glob
import os


# In[3]:


#1.Import health care dataset after downloading dataset from Kaggle
#read the dataset

df = pd.read_csv("/Users/44972/Desktop/BITSPilani/ML Assignment\heart.csv")
df.head()


# In[ ]:


#2.Extract X as all columns except the last column and Y as the last column.

from sklearn.preprocessing import StandardScaler

y = df.target
x = df.drop(columns = ['target'])


# In[35]:


#3.Visualize the dataset using any two appropriate graphs.
#3a.Histographic representation of Dataset

def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        
    fig.tight_layout()  
    plt.show()
draw_histograms(df,df.columns,6,3)


# In[36]:


#3b.Pairplot representation of Dataset

sn.pairplot(data=df,height=1)


# In[34]:


#4.Visualize the correlation between all the variables of a dataset. 

import matplotlib.pyplot as plt
import seaborn as sn

#correlation graph for the dataset
corr = df.corr()

plt.figure(figsize = (18,18))
sns.heatmap(corr, annot = True, cmap = 'coolwarm', vmin = -1, vmax=1)


# In[7]:


#5.Split the data into a training set and testing set.

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[37]:


#6.Implementation  10-fold cross-validation 

#Implementing cross validation
k = 10
kf = KFold(n_splits=k, random_state=None)
model = LogisticRegression(solver= 'liblinear')

acc_score = []

for train_index , test_index in kf.split(x):
    xtrain , xtest = x.iloc[train_index,:],x.iloc[test_index,:]
    ytrain , ytest = y[train_index] , y[test_index]
    
    model.fit(xtrain,ytrain)
    pred_values = model.predict(xtest)
    
    acc = accuracy_score(pred_values , ytest)
    acc_score.append(acc)
    
avg_acc_score = sum(acc_score)/k

print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))


# In[27]:


#7.Training a Logistic regression model for the dataset.
from sklearn.linear_model import LogisticRegression

def models(xtrain, xtest, ytrain, ytest):
    
    #logistic regression
    lrmodel = LogisticRegression(random_state = 0)
    lrmodel.fit(xtrain, ytrain)
    lrypred = lrmodel.predict(xtest)
    
    return lrypred


# In[28]:


lr = models (xtrain, xtest, ytrain, ytest)
lr


# In[11]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[31]:


#8.Computing the accuracy and confusion matrix.


ac = accuracy_score(ytest, lr)

lrcm = confusion_matrix(ytest, lr)

results = pd.DataFrame([['Logistic regression', ac,lrcm ]], columns = ['Model', 'Accuracy', 'Confusion Matrix'])


# In[32]:


print(results)


# In[33]:


#8a.Visualizing Confusion Matrix
import numpy as np

lrcm = confusion_matrix(ytest, lr)



models_list = [lrcm]
model_names = ['Logistic regression']

df_cm = pd.DataFrame(cl, index = (0,1), columns = (0,1))
plt.figure(figsize = (10,7))
sn.set(font_scale = 1.4)
sn.heatmap(df_cm, annot = True, fmt = 'g')
plt.title(model_names)

 


# In[48]:


ytrain.hist()


# In[45]:


ytest.hist()


# In[ ]:




