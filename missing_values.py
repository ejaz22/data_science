# for pandas tricks
https://nbviewer.jupyter.org/github/justmarkham/pandas-videos/blob/master/top_25_pandas_tricks.ipynb

   
# pandas top 10 categories with count
df['a'].value_counts().nlargest(10)

# use numpy.r_ to concanecate indices
df.iloc[:,np.r_[0,1,51:102]]

# create a column with missing value
df['Col'] = np.nan
df['Date']= pd.NaT

# find which column has missing data
pd.isnull(df).any()


# count NaN
df.isnull()                  #DataFrame of booleans
df.isnull().sum()            #List the NaN count for each column:
df.isnull().sum().sum()      #Count the total number of NaNs present in whole df

# sum
df[['a','b']].sum(axis=1)
df[['a','b']].values.sum()


# Counting Missing values: missing values are usually excluded by default
df['a'].value_counts()             # excludes missing values
df['a'].value_counts(dropna=False) # includes missing values


# Finding Missing Columns by using a boolean series to filter rows
df[df['a'].isnull()]        #only show rows where column_x is missing
df[df['a'].notnull()]       #only show rows where column_x is not missing


# fill NaN
df['a'].fillna(value='NA', inplace=True) 
df = df.fillna({
    'a': 'missing',
    'b': '99.999',
    'c': '999',
    'd': 'missing',
    'e': 'missing',
    'f': '99'
})

df['a'] = df['a'].replace(-77, np.NaN) # replace values with NaN



# Change all NaNs to None (useful before loading to a db)
df = df.where((pd.notnull(df)), None)


# drop missing values
df.dropna(inplace=True)             # drop rows if ANY values are missing, defaults to rows, with columns with axis=1
df.dropna(how=’all’, inplace=True)  # drop a row only if ALL values are missing
df.dropna(thresh=5)                 # drop rows less than 5 nan in a rwo
df = df.dropna(axis=0, subset=['Col']) # drop NaN from a particular col


# turn off the missing value filter, replaces NaN with /s
df = pd.read_csv(‘df.csv’, header=0, names=new_cols, na_filter=False)


# Conditional replacing Nan Value
df['new'] = np.where(pd.isnull(df['col1']),0,df['col1']) + df['col2'] # swap in 0 for df['col1'] cells that contain null
df['X'] = np.where(df['X'].isnull(),"BROKER",df['X']) # change null value only in a series
df['X'] = ('BROKER').where(df['X'].isnull())

# Conditional Mapping with np.select
conditions = [
            (df['ORIGINATOR']=='Capital One'),
            (df['ORIGINATOR']=='Meridian Broker')
            ]
df['BUSINESS_MIX'] = np.select(conditions,['CAPITAL ONE','MERIDIAN'],default='DIRECT')


# Univariate Selection
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)
print(fit.scores_)


# Feature Extraction with RFE
from sklearn.feature_selection import RFE
model = LogisticRegression()
rfe = RFE(model, 15)
fit = rfe.fit(X, y)
print fit.n_features_ , fit.support_ , fit.ranking_



# VarianceThreshold
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)
idxs = sel.get_support(indices=True)
np.array(X)[:, idxs]

# create dataframe with feature name after running SeletKBest
from sklearn.feature_selection import SelectKBest, f_classif
skb = SelectKBest(score_func=f_classif, k=5).fit_transform(X, y)
mask = skb.get_support()
new_features = X.columns[mask]

#other way to do th above task
cols = skb.get_support(indices=True)
# Create new dataframe with only desired columns, or overwrite existing
features_df_new = X[cols]


# RANDOM NUMBER AND STATISTICAL DISTRIBUTION
#===================================================================

import random
import numpy as np

# random.random() : This number is used to generate a float random number less than 1 and greater or equal to 0.

# seed() : this function maps a particular random number with the seed argument mentioned.
#All random numbers called after the seeded value returns the mapped number.

# Generate a list of number from 1 to 50
n = range(1,51)  # type list
n = np.arange(1,51)  # type ndarray

# PRNG - Generate a pseudo-random number between 0 and 1 
n = random.random()

# Pick a random number between 1 and 100.
random.randint(1, 100)    


# generate normal distribution
mu, sigma = 0.5, 0.1
s = np.random.normal(mu, sigma, 1000) #ndarray

# Binomial Distribution
from scipy.stats import binom
s = binom.rvs(size=10,n=20,p=0.8)

# Poisson distribution
from scipy.stats import poisson
s = poisson.rvs(mu=4, size=10000)

# Bernoulli Distribution
from scipy.stats import bernoulli
s = bernoulli.rvs(size=1000,p=0.6)


# Advance Regression kaggle
#!/usr/bin/env python
# coding: utf-8

# ## Regression Analysis

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import warnings


# In[2]:


# settings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='whitegrid')


# In[3]:


# read files
train = pd.read_csv(r'train.csv')
test = pd.read_csv(r'test.csv')
train_len = len(train)
# combine data and inspect data types
df = pd.concat([train, test], axis=0, ignore_index=True)
print('Data types are\n',df.dtypes.value_counts())


# In[4]:


# Describing Categorical Features
df.describe(include=['O']).T


# In[32]:


# Plot category - Univariate Analysis
plt.figure(figsize=(20,10)) 
sns.countplot(y="Neighborhood", data=df);


# In[5]:


# describe numeric features
df.describe().T


# In[3]:


# univariate analysis - chk distribution of target variable
df['SalePrice'].plot(kind='hist',bins=10)


# In[ ]:


train['SalePrice'].skew()


# In[ ]:


train['SalePrice'].kurt()


# In[3]:


#applying log transformation
df['SalePrice'] = np.log(df['SalePrice'])
df['SalePrice'].plot(kind='kde')


# In[ ]:


df['SalePrice'].skew()


# In[4]:


# Check Correlation for numeric features
corr = df.select_dtypes(include = [np.number]).corr()['SalePrice'].sort_values(ascending=False)
corr
#corr.style.background_gradient(cmap='coolwarm')


# In[4]:


# Null value analysis 
nulls = df.isnull().sum()
nulls = nulls[nulls > 0].sort_values(ascending=False)
pd.concat([nulls, nulls / df.shape[0]], axis=1, keys=['Missing','Missing_Ratio'])


# In[5]:


# drop feature that has no relevance such as ID or more thant 50% of null values
df.drop(['Id','PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage',
        'GarageFinish', 'GarageType', 'GarageQual', 'GarageYrBlt', 'GarageCond', 
        'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual','3SsnPorch',
         'MoSold','LowQualFinSF','YrSold','OverallCond','MSSubClass','GarageArea','1stFlrSF','FullBath',
         'YearBuilt','GarageYrBlt'
        ],axis=1,inplace=True)


# #### Categorical Value treatments

# In[6]:


# Convert to object
df['OverallQual'] = df['OverallQual'].astype('object')


# In[7]:


# Imputation with mode vlaue
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])
df['BsmtHalfBath'] = df['Functional'].fillna(df['Functional'].mode()[0])


# In[8]:


# One hot encoding
cat_features = df.select_dtypes(exclude=[np.number]).columns.to_list()
df_final = pd.get_dummies(df, columns=cat_features,drop_first=True)


# In[9]:


#df.dropna(inplace = True,axis=0)
df_final.fillna(df.mean(),inplace=True)


# In[10]:


# Build model parameter
import copy
final_train = copy.copy(df_final[:train_len])
final_test = copy.copy(df_final[train_len:])
y = final_train['SalePrice'] 
X = final_train.drop(['SalePrice'],axis=1)
y_test = final_test.drop(['SalePrice'],axis=1)


# In[17]:


# Create Sample Submission file
pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv(r'C:\Users\thw202\Desktop\kaggle\sample_submission.csv')
datasets = pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns= ['Id','SalePrice']
datasets.to_csv(r'C:\Users\thw202\Desktop\kaggle\sample_submission1.csv',index=False)


# In[11]:


# Initialize regesssion models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
seed = 44

# used as of now
lr = LinearRegression(n_jobs = -1)
lasso = Lasso(random_state = seed)
ridge = Ridge(random_state = seed)
elnt = ElasticNet(random_state = seed)
svr = SVR()
rf = RandomForestRegressor(n_jobs = -1, random_state = seed)
knn = KNeighborsRegressor(n_jobs= -1)
dt = DecisionTreeRegressor(random_state = seed)
#gb = GradientBoostingRegressor(random_state = seed)

# others
kr = KernelRidge()
pls = PLSRegression()
et = ExtraTreesRegressor(n_jobs = -1, random_state = seed)
ab = AdaBoostRegressor(random_state = seed)


# In[ ]:


# Hyperparameter tuning - Ridge Regression
ridge_param_grid = {'alpha':[0.5, 2.5, 3.3, 5, 5.5, 7, 9, 9.5, 9.52, 9.64, 9.7, 9.8, 9.9, 10, 10.5,10.62,10.85, 20, 30],
                    'random_state':[seed]}
grid = GridSearchCV(ridge, param_grid=ridge_param_grid, cv = 10, verbose = 1, scoring = 'neg_mean_squared_error', n_jobs = -1)
grid.fit(X, y)
best_params = grid.best_params_ 
best_score = np.round(np.sqrt(-1 * grid.best_score_), 5)
print (best_params, best_score, grid.feature_importances_)
#scores = cross_val_score(grid, X, y, scoring='accuracy',cv=5)


# In[ ]:


# perform ridge regression
op_ridge = Ridge(**grid.best_params_)
op_ridge.fit(X,y)
y_pred = op_ridge.predict(y_test)
np.sqrt(mean_squared_error(y[:-1], y_pred))


# In[ ]:


# Hyperparameter tuning Lasso
lasso_param_grid = {'alpha':[0.0001, 0.0002, 0.00025, 0.0003, 0.00031, 0.00032, 0.00033, 0.00034, 0.00035, 0.00036, 0.00037, 0.00038, 
                            0.0004, 0.00045, 0.0005, 0.00055, 0.0006, 0.0008,  0.001, 0.002, 0.005, 0.007, 0.008, 0.01],
                   'random_state':[seed]}
grid = GridSearchCV(lasso, param_grid=lasso_param_grid, cv = 10, verbose = 1, scoring = 'neg_mean_squared_error', n_jobs = -1)
grid.fit(X, y)
best_score = np.round(np.sqrt(-1 * grid.best_score_), 5)
print (grid.best_params_, best_score)


# In[ ]:


# perform Lasso regression
op_lasso = Ridge(**grid.best_params_)
op_lasso.fit(X,y)
y_pred = op_lasso.predict(y_test)
np.sqrt(mean_squared_error(y[:-1], y_pred))


# In[ ]:


grid.best_params_


# In[ ]:


# Hyperparameter tuning - Elastic Net
elastic_params_grid = {'alpha': [0.0001,0.0002, 0.0003, 0.01,0.1,2], 
                       'l1_ratio': [0.2, 0.85, 0.95,0.98,10],
                       'random_state':[seed]}
grid = GridSearchCV(elnt, param_grid=elastic_params_grid, cv = 10, verbose = 1, scoring = 'neg_mean_squared_error', n_jobs = -1)
grid.fit(X, y)
best_params = grid.best_params_ 
best_score = np.round(np.sqrt(-1 * grid.best_score_), 5)
print (best_params, best_score)


# In[ ]:


# Hyperparameter tuning - SVR
svr_params_grid = {'kernel':['linear', 'poly', 'rbf'],
                   'C':[2,4,5],
                   'gamma':[0.01,0.001,0.0001]}
grid = GridSearchCV(svr, param_grid=svr_params_grid, cv = 10, verbose = 1, scoring = 'neg_mean_squared_error', n_jobs = -1)
grid.fit(X, y)
best_params = grid.best_params_ 
best_score = np.round(np.sqrt(-1 * grid.best_score_), 5)
print (best_params, best_score)


# In[ ]:


# Hyperparameter tuning - Random Forest Regressor
rf_params_grid = {'n_estimators':[1,5,50,100],
                   'max_depth':[1,2],
                   'min_samples_split':[3,4],
                   'min_samples_leaf':[2,4],
                   'random_state':[seed]}
grid = GridSearchCV(rf, param_grid=rf_params_grid, cv = 10, verbose = 1, scoring = 'neg_mean_squared_error', n_jobs = -1)
grid.fit(X, y)
best_params = grid.best_params_ 
best_score = np.round(np.sqrt(-1 * grid.best_score_), 5)
print (best_params, best_score)


# In[ ]:


import sklearn
sorted(sklearn.metrics.SCORERS.keys())


# In[23]:


# Fit GB Regressor model
params = {'n_estimators': 1250, 'max_depth': 4, 'min_samples_split': 4,
          'learning_rate': 0.025, 'loss': 'ls'}

gb = GradientBoostingRegressor(**params).fit(X,y)
gb.score(X,y)

#cross_val_score(gb,X,y,scoring='neg_mean_squared_error')


# In[15]:


y_pred = gb.predict(y_test)
np.sqrt(mean_squared_error(y[:-1], y_pred))


# In[22]:


# convert to log transformed variable to exponential
y_pred = np.exp(y_pred) 


# In[ ]:


# Hypertuning - Gradient Boosting Regressor
gb_param = {'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
            'n_estimators':[100,250,500,750,1000,1250,1500,1750],
            'min_samples_leaf':[3],
            'max_depth':[6],
            'max_features':[1.0]}

grid = GridSearchCV(gb, param_grid = gb_param, cv=10, n_jobs=-1)
grid.fit(X, y)
best_score = np.round(np.sqrt(-1 * grid.best_score_), 5)
print (grid.best_params_ , best_score)

# nested cross validation
#scores = cross_val_score(gs, X_train, y_train, scoring='accuracy',cv=5)


# In[14]:


# perform KNN Regression
knn = KNeighborsRegressor(n_neighbors=5,weights='uniform').fit(X,y)
knn.score(X,y)
#y_pred = knn.predict(y_test)


# In[15]:


# Baysian Regression
breg = linear_model.BayesianRidge().fit(X,y)
breg.score(X,y)
#y_pred = breg.predict(y_test)


# In[18]:


# Decision Tree Regression
dt = DecisionTreeRegressor(max_depth=5).fit(X,y)
dt.score(X,y)
#from sklearn.metrics import r2_score
#r2_score()
#y_pred = dt.predict(y_test)


# In[25]:


# residual plot
import matplotlib
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
preds = pd.DataFrame({"preds":gb.predict(X), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
plt.title("Residual plot")


# In[ ]:





