# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:24:53 2020

@author: maria
"""

import pandas as pd
import numpy as np

orders0 = pd.read_table('http://bit.ly/chiporders')
ufo0 = pd.read_csv('http://bit.ly/uforeports', sep = ',')
movies0 = pd.read_csv('http://bit.ly/imdbratings')
drinks0 = pd.read_csv('http://bit.ly/drinksbycountry')

# Go from a SERIES to a DATA FRAME
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
#Dict helps
pd.DataFrame({ 'City name': city_names, 'Population': population })
#It allows to quickly study the distribution of an only collumn
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.hist('housing_median_age')

#1 How do I read tabular data file into pandas
#pd.read_table('data/chipotle.tsv')
orders = orders0
orders.head() # returns first rows (default 5)
orders.tail() # returns last n rows (default 5)

user_cols = ['user-id', 'age', 'gender', 'occupation', 'zip_code']
orders1 = pd.read_table('http://bit.ly/movieusers', sep = '|', header = None, names = user_cols)

#2 How to select pandas series from data frame
ufo = ufo0
# CSV file, default is separation with comma 
type(ufo)
ufo['City']
type(ufo['City'])
# ufo.City (Shows all data)
# ufo['Colors Reported']
# ufo.shape
# TO CREATE A NEW SERIES IIN THE DATA FRAME
ufo.City + ', ' + ufo.State
ufo['Location'] = ufo.City + ', ' + ufo.State

#3 why do some methods/attributes have parenthesis or not
movies = movies0
movies.describe() #GIVES A DESCRIPTIVE VIEW ON NUMERICAL SERIES OF DATA! MEDIAN ETC
movies.shape
movies.dtypes
""" Movies is a data frame (type), so there are certain methods
(with parenthesis) and attributes (withour parenthesis)
Methods - actions; attributes - descriptions; MAKES SENSE """
movies.describe(include = ['object'])

#4 How do I rename columns in a panda DataFrame
""" Use ufo """
ufo = ufo0
varname = ufo.columns
ufo.rename(columns = {'Colors Reported' : 'Colors_Reported', 'Shape Reported' : 'Shape_Reported'}, inplace = True)
newvarname = list(ufo.columns)
ufo_cols = ['city', 'colors reported', 'shape reported', 'state', 'time', 'location']
ufo.columns = ufo_cols
# ufo = pd.read_csv('http://bit.ly/uforeports', names = ufo_cols, header = 0) # if headers is not added the previous would stay
ufo.columns = ufo.columns.str.replace(' ', '_')
# with one line replace space in header with underscore

# 5 How to remove columns from panda DataFrame
""" Use ufo """
ufo = ufo0
ufo.drop('colors_reported', axis = 1, inplace =True)
# axis is a direction; 0 is a row; 1 is a columns!
# 1 is drop a column
# inplace = True, I wat the operation to happen in place
# Not possible to drop multiple at the same time!
ufo.drop(2, axis = 0, inplace = True)
ufo.drop([0,1], axis = 0, inplace = True)
ufo.drop(range(3, 5), axis = 0, inplace = True)

# 6 How do I sort a pandas DataFrame or Series?
""" Use movies..."""
movies.title.sort_values()
movies['title'].sort_values()
# Different 
movies.sort_values('title')
movies.sort_values('duration', ascending = False)

# 7 How do I filter rows of columns of pandas by column value
movies = movies0
booleans = []
for length in movies.duration:
    if length >= 200:
        booleans.append(True)
    else: 
        booleans.append(False)

is_long = movies.duration >= 200
# CREATES THE SAME AS THE LOOP 
is_long = pd.Series(booleans)
movies[is_long] # Just show data that is longer or equal to 200 mintues!!!

#SIMPLER VERSION !!!
movies[movies.duration >= 200]
# movies[movies.duration >= 200].genre SHOWS THE GENRES OF THESE MOVIES WITH MORE THAN 200 MINUTES

# 8 How do I apply multiple filter criteria to a pandas DataFrame
""" Use movies """
movies = movies0
len(movies[(movies.duration >= 200) & (movies.genre == 'Drama')])
len(movies[(movies.duration >= 200) | (movies.genre == 'Drama')])
# | OR & AND
movies[movies.genre.isin(['Crime', 'Drama', 'Acition'])]

# 9 How to readin csv only two collumns; what's the fastest
# Read only 2 columns
ufo = pd.read_csv('http://bit.ly/uforeports', usecols = [0,4], nrows = 3)
# or usecols = ['City', 'Time']
ufo.shape

for index, row in ufo.iterrows():
    print(index, row.City)

#Best way to drop all non numerical data
drinks = drinks0
drinks.dtypes # gives you type of each set of data (each column data)
drinks.select_dtypes(include = [np.number]).dtypes

drinks.describe(include = ['object'])

# 10 How do I use the axis parameter in pandas
drinks.drop('continent', axis = 1).head() # Drops columns Continent
drinks.drop(2, axis = 0).head() # drops row number 2

drinks.mean() # default is column, axis 0; for each row axis = 1
drinks.mean(axis = 'index').shape
drinks.mean(axis = 'columns').shape

# 11 How do I change the data type of panda Series?
drinks = drinks0
drinks.dtypes # gives you type of each set of data (each column data)

drinks['beer_servings'].astype(float)
#SAME drinks.beer_servings.astype(float)
# CHANGE TYPES OF TIME SERIES DURING READING PROCESS
drinks = pd.read_csv('http://bit.ly/drinksbycountry', dtype = {'beer_Servings':float})

orders = orders0
#price data is recognised as object cause the dollar sine is not recognised
orders.item_price.str.replace('$', ' ').astype(float).mean()
orders.item_name.str.contains('Chicken').astype(int).head()
# 0 or 1 if there is chicken on does rows

# 12 When should I use grouby in pandas
drinks = drinks0
drinks.beer_servings.mean()
drinks.groupby('continent').beer_servings.mean()
drinks[drinks.continent == 'Africa'].mean()
drinks[drinks.continent == 'Africa'].beer_servings.mean()
# Use groupby anytime you want to analyse data series per category !!!
drinks.groupby('continent').mean()
drinks.groupby('continent').beer_servings.agg(['count', 'min', 'max'])
drinks.groupby('continent').mean().plot(kind ='bar')

# 13 -> 25 top trics!
# Show installed version
pd.__version__
# Create an examplde DataFrame
pd.DataFrame(np.random.rand(4,8))
pd.DataFrame(np.random.rand(3,8), columns = list('abcdefgh'))
df = pd.DataFrame({'Col one':[100, 200], 'Col two':[300, 400]})
#Rename columns
df = df.rename({'col one':'col_one', 'col two':'col_two'}, axis = 'columns')
df.columns = ['col_one', 'col_two'] # overwrite
df.columns = df.columns.str.replace(' ', '_')
df.add_prefix('X__')
df.add_suffix('__Y')
# Reverse order
drinks= drinks0
drinks.loc[::-1].head()
drinks.loc[::-1].reset_index(drop = True)
drinks.loc[:, ::-1].head()

drinks.select_dtypes(include = 'number')
#drinks.select_dtypes(remove = 'number')

#Convert DataFrame types of data
# df.types
# df.astype({'col_one':'float', 'col_two':'float'}).dtypes
## df = df.apply(pd.to_numeric(df.col_three, errors = 'coerce').fillna(0)

# DATA FRAME AND MEMORY
drinks.info(memory_usage = 'deep')
# Only read the columns you need
# Category data type... convert everything you can

#Build a Data Frame from multiple files (row = wise)
from glob import glob
# Looks for all files in data directory with the name of stocks
stock_files = sorted(glob('DATA/STOCKS*.CSV'))
# It returns a list with all 3 files
# pd.concat((pd.read_csv(file) for file in stock_files), ignore_index = True)

# 10 Builda Dataframe from multiple files (column wise)
# drink_files = sortes(glob('data/drinks*.csv'))
# pd.concat((pd.read_csv(file) for file in drink_files), axis = 'columns')

# CREATE A DATA FRAME FROM CLIPBOARD
# Type on excel what you want and copy, then
# df = pd.read_clipboard()

# 12 SPLIT A DATAFRAME INTO TWO RANDOM SUBSETS
len(movies)
movies_1 = movies.sample(frac = 0.75, random_state = 1234)
movies_2 = movies.drop(movies_1.index)
movies_1.index.sort_values()
movies_2.index.sort_values()

# Handling missing values
ufo = ufo0
ufo.isna().sum()
ufo.dropna(axis = 'columns')
ufo.dropna(thresh = len(ufo)*0.9, axis = 'columns')

# Aggregate by multiple functions
orders.groupby('order_id').item_price.agg(['sum', 'count'])

# Combine the previous with the DataFrame
# total_price = orders.groupby('order_id').item_price.transform('sum')
# orders['total_price'] = total_price

# 20 Select a slice of rows and columns
# titcanic.describe().loc['min':'max', 'columnn1':'column2']

# titanic.groupby(['Sex', 'Pclass']).Survived.mean()

# Creat Pivot table
# titanic.pivot_table(index = 'Sex', columns='Pclass', values = 'Survived', aggfunc='count', margins = True)

# Convert continuous data into categorical data
# pd.cut(titanic.Age, bins = [0, 18, 25, 99]), labels = ['child', 'young adult', 'adult'])

# Change display options!! 
# pd.set_options('display.float_format', '{:.2f}'.format)

# STYLE A DATAFRAME
# format_dict = {'Date':'{:%m/%d/%y}', 'Close':'${:.2f}', 'Volume':'{:,}'}
# stocks.style.format(format_dict).hide_index()
# stocks.style.format(format_dict).hide_index().highlight_min('Close', color = 'red').highlight_max('Close', color = 'lightgreen')
# stocks.style.format(format_dict).hide_index().background_gradient(subset = 'Volume', cmap = 'Blues')
# stocks.style.format(format_dict).hide_index().bar('Volume', color = 'lightblue', align = 'zero').set_captation('Stock Prices from October 2016')

# import pandas_profiling
# pandas_profiling.ProfileReport(drinks)

# How do I find and remove duplicate rows in pandas
# data1.duplicated() - when entire rows are duplicated
# data1.settlement_period.duplicated()
# users.duplicates(subset = ['age', 'zip_code'])
# users.drop_duplicates(subset = ['age', 'zip_code'])

# HOW DO I EXPLORE A PANDAS SERIES
movies = movies0
movies.genre.describe()
movies.genre.value_counts()
movies.genre.value_counts(normalize = True) # type series
movies.genre.unique()
movies.genre.nunique()
pd.crosstab(movies.genre, movies.content_rating)
movies.duration.describe()
movies.duration.mean()
movies.duration.value_counts()

# HOW to handle missing values in pandas
ufo0 = pd.read_csv('http://bit.ly/uforeports', sep = ',')
ufo0.isnull().tail() # Shows True or False for missing values or not
ufo0.notnull().tail() # Shows False for the missing values
ufo0.isnull().sum() # Shows total NaN members for each column
pd.Series([True, False, True]).sum() # True - 1; False - 0
# DEFAULTS IS AXIS = 1
ufo0[ufo0.City.isnull()] # SHOWS ONLY ROWS WHERE city HAS NaN!!!
ufo0.dropna(how = 'any').shape # DROP ALL ROWS WHERE ANY COLLUMN HAS NaN! any
# (it's a temporary)
ufo0.dropna(how = 'all').shape # Only if all have NaN - which will not happen because two columns have 0 NaN
ufo0.dropna(subset = ['City', 'Shape Reported'], how = 'all') # Only considers drop if all of the columns especified has NaN
ufo0['Shape Reported'].value_counts()
ufo0['Shape Reported'].value_counts(dropna = False) # It will also show NaN rows
ufo0['Shape Reported'].fillna(value = 'VARIOUS', inplace = True) # fill in Nan values with VARIOUS

# What do I need to know about the pandas index?
drinks = drinks0
# Index is always there, is not optional
# Index or Columns are not part of the data frame!!
drinks[drinks.continent == 'South America']
# By setting the index with some meaningful data for us we can select data more easily
drinks.set_index('country', inplace = True)
# FIND data
drinks.loc['Brazil','beer_servings']
drinks.index.name = None
drinks.index.name = 'country'
# To reset index
drinks.reset_index(inplace = True) # but before remove again the index, otherwise the column from the previous new index will be called index
# Trick
drinks.describe().loc['25%', 'beer_servings']

# How to use index part 2
drinks.set_index('country', inplace = True)
drinks.continent.value_counts()
drinks.continent.value_counts().values
drinks.continent.value_counts()['Africa']
drinks.continent.value_counts().sort_values()

# Make small data set
people = pd.Series([300000, 50000], index = ['Albania', 'Andorra'], name = 'Population')
drinks.beer_servings * people
pd.concat([drinks, people], axis = 1) # axis 1 says data side by side

# How do I select multiple rows and columns form pandas DataFrame?
# ufo.loc[] filters and selects data by label (index or columns)
# structures is ufo.loc[rows I want, columns I want]
ufo.loc[5, :]
ufo.loc[[5, 6, 7], :]
ufo.loc[:, 'city']
ufo.loc[:, ['city', 'state']]
ufo.head(3).drop('time', axis =1)
ufo[ufo.city == 'Oakland']
ufo.loc[ufo.city == 'Oakland', :]
ufo.iloc[:, [0,3]] # filter data by integer position
ufo[['city', 'state']]
ufo[5:6]
# ix allows to blend between label and integer
# drinks.ix['Albania', 0]

# HOW do work with data times on pandas
ufo.time.str.slice(-5, -3).astype(int).head()
ufo['time'] = pd.to_datetime(ufo.time)
ufo.time.dt.weekday_name # It will get you the week of the day
ufo.time.dt.dayofyear
ts = pd.to_datetime('1/1/1999')
ufo.loc[ufo.time >= ts, :]

# HOW do I create a pandas DataFrame from another object
# Create a data frame from dic
df = pd.DataFrame({'id': [100, 101, 102], 'color': ['red', 'blue', 'red']}, columns = ['id', 'color'], index = ['a', 'b', 'c'])
pd.DataFrame([[100, 'red'], [101, 'blue'], [102, 'red']], columns = ['id', 'color'])
arr = np.random.rand(4, 2)
pd.DataFrame(arr, columns = ['one', 'two'])
pd.DataFrame({'student': np.arange(100, 110, 1), 'test':np.random.randint(60, 101, 10)}).set_index('student')
s = pd.Series(['round', 'square'], index=['c', 'b'], name = 'shape')
pd.concat([df, s], axis = 1 )

# HOW do I apply a function to a pandas Series or DataFrame
train = pd.read_csv('http://bit.ly/kaggletrain')
# Use map to set a Serios into different values
train['Sex_num'] = train.Sex.map({'female': 0 , 'male': 1})
train.loc[0:4, ['Sex', 'Sex_num']]
# Apply a funtion into a specific Series
train['Name_length'] = train.Name.apply(len)
train.loc[0:4, ['Name', 'Name_length']]
train['Fair_ceil'] = train.Fare.apply(np.ceil) # ROUNDED UP each value
train.loc[0:4, ['Fare', 'Fare_ceil']]
train.Name.str.split(',').head() # Separates strings in lists devided by the previous commas
def get_element(my_list, position):
        return my_list[position]
train.Name.str.split(',').apply(get_element, position = 0).head()
# SAME AS PREVIOUS APPLY METHOD for Series
train.Name.str.split(',').apply(lambda x: x[0]).head()
# SHOWS THE MAX OF EACH ROW!
drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis = 0)
# TO know the collumn with higher value
drinks.loc[:, 'beer_servings':'wine_servings'].apply(np.argmax, axis = 0)
# APPLYMAP applies a funtion to every element in the DataFrame
drinks.loc[:, 'beer_servings':'wine_servings'].apply(float)

# How to select multi index
stocks = pd.read_csv('http://bit.ly/smallstocks')
stocks.index
stocks.groupby('Symbol').Close.mean()
# Series with a multy index - it adds another dimension to your data
ser = stocks.groupby(['Symbol', 'Date']).Close.mean()
ser.index
ser.unstack() # MAKES IT AS A DATAFRAME
# SAME 
df = stocks.pivot_table(values = 'Close', index = 'Symbol', columns = 'Date')
ser.loc['AAPL'] # Just select the outer label! to get all data with AAPL as index
ser.loc['AAPL', '2016-19-03']
ser.loc[:, '2016-10-03']

stocks.set_index(['Symbol', 'Date'], inplace = True)
# FIRST SORT IT
stocks.sort_index(inplace = True) # Sorts outer index first and after outer
stocks.loc[('AAPL', '2016-10-03'), :] # Say what rows you want  and the collumns - all (:)
stocks[('AAPL', '2016-10-03'), 'Close']
stocks[(['AAPL','MSFT'], '2016-10-03'), :]
# MULTIPLE ^ STOCKS SO SQUARED BRACKETS
stocks.loc[(slice(None), ['2019-10-03', '2016-10-04']), :]
# Slice(None) reqiured as two indexes!

# MERGE TWO DATA SETS WITH THE SAME INDEX !!! SIDE BY SIDE
# both = pd.merge(close, volume, left_index = True, right_index = True)
# both.reset_index()

# MERGING AND JOINNING

# df1.append(df2) : stacking vertically
# pd.concat([df1, df2]) : STACK VERT AND
# df1.join(df2): inner/outer/left/right joins or indexes
# pd.merge([df1, df2]) : many joins on multiple columns

# ratings.movie_id.nunique() shows the unic values in specific column
# ratings.loc[ratings.movie_id == 1, :]
# movie_ratings = pd.merge(movies, ratings)
# if there are repeated columns they merge

# What if columns do not have the same name !!!
# pd.merge(movies, ratings, left_on = 'm_id', right_on = 'movie_id') - they merge!!!
# pd.merge(movies, ratings, left_index = True, right_on = 'movie_id')

# INNER JOIN - just join rows if there is common info
# pd.merge(A, B, how = 'inner')
# OUTER - all rows get included regardless of common info

# 4 TIME SAVING TRICKS IN PANDAS
# new: creates a datetime column from the entire DataFrame
df = pd.DataFrame([[12, 25, 2017, 10], [1, 15, 2018, 11]], columns = ['month', 'day', 'year', 'hour'])
pd.to_datetime(df)
pd.to_datetime(df[['month', 'day', 'year']])

# Create a category column during file reading
# 'Category' type collumn... sometimes useful, to make it faster as well
# OR
drinks = pd.read_csv('http://bit/ly/drinksbycountry', dtype = {'continent':'category'})

# Convert the data type of multiple columns at once !
drinks = drinks.astype({'beer_servings':'float', 'spirit_servings': 'float'})
# Converts all types in dic at the same time...

# Apply miultiple aggregations on a Series or DataFrame
drinks.groupby('continent').beer_servings.agg(['mean', 'min', 'max'])
drinks.beer_servings.agg(['mean', 'min', 'max'])
drinks.agg(['mean', 'min', 'max']) # Do aggregated functioons at the same time in the whole DataFrame



