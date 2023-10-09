#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction of house prices in Ames, Iowa, using the random forest algorithm

"""


# SETUP



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt



# DATA LOADING

"""We use the Ames Iowa Housing dataset, which is a public dataset documenting
prices of sold houses in Ames, Iowa. The dataset is available - among other place - at:
    https://www.kaggle.com/datasets/marcopale/housing
""" 
# read in training data
train_path = '/Users/pthompson/Downloads/house-prices-advanced-regression-techniques/train.csv'
train_data = pd.read_csv(train_path)
train_data.head()

train_data.describe()


y_train = train_data.loc[:,'SalePrice']
X_train = train_data.drop(labels=['SalePrice','Id'],axis='columns')

print(y_train.shape)
print(X_train.shape)



# VISUALIZATIONS

plt.title('Histogram of Target (Sale Price)')
sns.histplot(y_train)

plt.title('KDE plot of Target (Sale Price) by House Style')
sns.kdeplot(data=train_data,x='SalePrice',hue='HouseStyle',shade=True)

plt.title('Scatter plot of Target (Sale Price) and Lot Area, color-coded by Building Type')
sns.scatterplot(data=train_data,x='LotArea',y='SalePrice',hue='BldgType')



# PREPROCESSING
# dropping columns with too many missing values. The others will be imputed
missing_val_count_by_column = (X_train.isnull().sum())
too_many_missing = list(missing_val_count_by_column[missing_val_count_by_column > 0].index)
missing_y = (y_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 100])
print(missing_y)
print("too many missings in: ",too_many_missing)


X_train = X_train.drop(labels=too_many_missing,axis='columns')

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64','float64']]
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'object' and
                   X_train[cname].nunique() < 10]

print("no of numerical columns: ",len(numerical_cols))
print("numerical columns: ",numerical_cols)
print("no of categorical columns: ",len(categorical_cols))
print("categorical columns: ",categorical_cols)



# MODELING
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error


# set up preprocessing and modeling pipeline parts
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown ='ignore'))
])

preprocessor = ColumnTransformer(
transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

randomforest = RandomForestRegressor(random_state=0)

# build pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('randomforest',randomforest)
])


# fit model
my_pipeline.fit(X_train,y_train)


# MODEL EVALUATION
train_preds = my_pipeline.predict(X_train)

mean_absolute_error(y_train,train_preds)



#PREDICTION

test_path = '/Users/pthompson/Downloads/house-prices-advanced-regression-techniques/test.csv'
test_data = pd.read_csv(test_path)
X_test = test_data.drop(labels='Id',axis='columns')
X_test = X_test.drop(labels=too_many_missing,axis='columns')

test_preds = my_pipeline.predict(test_data)









# OPTIONAL: HYPERPARAMETER TUNING

param_grid = {
    'randomforest__n_estimators': [100,500,1000,2000],
    'randomforest__max_depth': [5,15],
    'randomforest__min_samples_split': [5,10,15,20,25,30]
}

search = GridSearchCV(my_pipeline,
                     param_grid,
                     n_jobs = 4,
                     cv = 5,
                     scoring = 'neg_mean_squared_error')

search.fit(X_train,y_train)


print(search.best_params_)
print(search.score(X_train, y_train))



final_randomforest = RandomForestRegressor(n_estimators = 500,
                                            max_depth = 15,
                                            min_samples_split = 5,
                                            random_state=0)
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('randomforest', final_randomforest)
                             ])

final_pipeline.fit(X_train, y_train)


