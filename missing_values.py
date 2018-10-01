
# MISSING VALUES
===================================================

# find missing values
df.isnull()                  #DataFrame of booleans
df.isnull().sum()            #List the NaN count for each column:
df.isnull().sum().sum()      #Count the total number of NaNs present in whole df


# Finding Missing Columns by using a boolean series to filter rows
df[df['x'].isnull()]        #only show rows where column_x is missing
df[df['x'].notnull()]       #only show rows where column_x is not missing


# fill in missing values
df.column_x.fillna(value='NA', inplace=True) 


# create a column with missing value
df['location'] = np.nan


# drop missing values
df.dropna(inplace=True)             # drop rows if ANY values are missing, defaults to rows, with columns with axis=1
df.dropna(how=’all’, inplace=True)  # drop a row only if ALL values are missing
df.dropna(thresh=5)                 # drop rows less than 5 nan in a rwo


# Counting Missing values: missing values are usually excluded by default
df.column_x.value_counts()             # excludes missing values
df.column_x.value_counts(dropna=False) # includes missing values


# turn off the missing value filter, replaces NaN with /s
df = pd.read_csv(‘df.csv’, header=0, names=new_cols, na_filter=False)


# Clean up missing values in multiple DataFrame columns
df = df.fillna({
    'col1': 'missing',
    'col2': '99.999',
    'col3': '999',
    'col4': 'missing',
    'col5': 'missing',
    'col6': '99'
})


# Conditional replacing Nan Value
df['new'] = np.where(pd.isnull(df['col1']),0,df['col1']) + df['col2'] # swap in 0 for df['col1'] cells that contain null
df['X'] = np.where(df['X'].isnull(),"BROKER",df['X']) # change null value only in a series
