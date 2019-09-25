# MISSING VALUES
===================================================

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
df['column_x'].value_counts()             # excludes missing values
df.column_x.value_counts(dropna=False) # includes missing values


# Finding Missing Columns by using a boolean series to filter rows
df[df['x'].isnull()]        #only show rows where column_x is missing
df[df['x'].notnull()]       #only show rows where column_x is not missing


# fill NaN
df['col1'].fillna(value='NA', inplace=True) 
df = df.fillna({
    'col1': 'missing',
    'col2': '99.999',
    'col3': '999',
    'col4': 'missing',
    'col5': 'missing',
    'col6': '99'
})

df['col1'] = df['col1'].replace(-77, np.NaN) # replace values with NaN



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
