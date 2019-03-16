#!/usr/bin/env python
# coding: utf-8

# In[158]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[95]:


pwd


# In[96]:


cd "Desktop/mch learning/"


# In[202]:


df=pd.read_csv('train.csv')
dt=pd.read_csv('test.csv')
dg=pd.read_csv('gender_submission.csv')


# In[98]:


df.head(10)


# In[99]:


df.columns


# In[100]:


df.shape


# In[101]:


df.info()


# In[102]:


df['Embarked'].value_counts()


# In[ ]:





# In[183]:


df.Embarked.fillna('S')
dt.Embarked.fillna('S')
x=1


# In[104]:


import seaborn as sns
sns.distplot(df.Fare)


# In[105]:


sns.distplot(df[df['Age'].notnull()]['Age'])


# In[106]:


df.Age.describe()


# In[184]:


#mean can not be use because std is high
# we will use interpolation method
df.Age=df.Age.interpolate()
dt.Age=dt.Age.interpolate()
x=1


# In[108]:


df.Age.describe()


# In[109]:


sns.distplot(df.Age)


# In[185]:


# distributoion does note change
df=df.drop('Cabin',axis=1)
dt=dt.drop('Cabin',axis=1)


# In[186]:


df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
dt = dt.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
df.head()


# In[112]:


sns.countplot(x='Pclass',data=df)


# In[113]:


sns.set(style="whitegrid")
g = sns.PairGrid(data=df, x_vars=['Pclass'], y_vars='Survived', size=5)
g.map(sns.pointplot, color=sns.xkcd_rgb["plum"])
g.set(ylim=(0, 1))


# In[114]:


from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=[15,15])
plt.show()


# In[115]:


h = sns.PairGrid(data=df, x_vars=['Sex'], y_vars='Survived', size=5)
h.map(sns.pointplot)
h.set(ylim=(0, 1))


# In[187]:


#we create a in feature is_child(below 15)
dt['is_child'] = dt['Age'].apply(lambda x: 1 if x <= 15 else 0)
df['is_child'] = df['Age'].apply(lambda x: 1 if x <= 15 else 0)


# In[117]:


i = sns.PairGrid(data=df, x_vars=['is_child'], y_vars='Survived', size=5)
i.map(sns.pointplot)
i.set(ylim=(0, 1))


# In[118]:


df.head()


# In[119]:


df.columns


# In[ ]:





# In[188]:


dt['family'] = dt['SibSp'] + dt['Parch']
dt = dt.drop(['SibSp', 'Parch'], axis=1)
df['family'] = df['SibSp'] + df['Parch']
df = df.drop(['SibSp', 'Parch'], axis=1)


# In[192]:


df.head()


# In[193]:


dt['is_alone'] = dt['family'].apply(lambda x: 1 if x == 0 else 0)
df['is_alone'] = df['family'].apply(lambda x: 1 if x == 0 else 0)


# In[123]:


df.columns


# In[124]:


df.Pclass.value_counts()


# In[194]:


dt = pd.get_dummies(dt)
df = pd.get_dummies(df)


# In[156]:


df


# In[151]:


y=df['Survived']


# In[152]:


X=df.drop(['Survived'],axis=1)


# In[165]:


import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[155]:


from sklearn.model_selection import cross_val_score
print(((cross_val_score(DecisionTreeClassifier(),X,y,cv=7))))


# In[160]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[161]:


train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75, test_size=0.25, stratify=y)


# In[162]:


def build_classifier(model):
    classifier = model()
    classifier.fit(train_X, train_y)
    print(classifier.score(test_X, test_y))
    return classifier


# In[163]:


decision_tree = build_classifier(GradientBoostingClassifier)


# In[168]:


from sklearn.model_selection import cross_val_score
print(np.mean((cross_val_score(LinearRegression(),X,y,cv=7))))


# In[170]:


from sklearn.model_selection import cross_val_score
print(np.mean((cross_val_score(KNeighborsClassifier(),X,y,cv=7))))


# In[172]:


from sklearn.model_selection import cross_val_score
print(np.mean((cross_val_score(GaussianNB(),X,y,cv=7))))


# In[173]:


from sklearn.model_selection import cross_val_score
print(np.mean((cross_val_score(SVC(gamma='auto'),X,y,cv=7))))


# In[178]:


from sklearn.model_selection import cross_val_score
print(np.mean((cross_val_score(DecisionTreeClassifier(),X,y,cv=7))))


# In[179]:


regressor=DecisionTreeClassifier()
regressor.fit(X,y)


# In[198]:


dt.Fare=dt.Fare.interpolate()
dt.info()


# In[200]:


db=regressor.predict(dt)


# In[203]:


dg.Survived=db


# In[205]:


dg.to_csv('gendersubmission.csv',index=False)


# In[206]:


pwd


# In[ ]:




