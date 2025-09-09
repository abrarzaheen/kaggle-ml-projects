#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Read data
df = pd.read_csv(r'data\train.csv')

#Separate features and target
X = df.drop(columns=['Id', 'SalePrice'])
y = df['SalePrice'].values

#Get column names from the feature matrix X (not the original df)
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()


#Special categorical columns where NaN means "None" or absence
none_fill = [
    "PoolQC","MiscFeature","Alley","Fence","FireplaceQu",
    "GarageType","GarageFinish","GarageQual","GarageCond",
    "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
    "MasVnrType"
]


# Filter none_fill to only include columns that actually exist in cat_cols
none_fill = [col for col in none_fill if col in cat_cols]

# Separate the columns where NaN means missing (To be filled with mode)
mode_fill = [col for col in cat_cols if col not in none_fill]

# All numeric columns (we'll handle missing values in the pipeline)
num_fill = num_cols


#Pipeline for applying multiple transformations on multiple columns at the same time
none_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant',fill_value='None')),
    ('encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False))
])

mode_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False))
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# Create preprocessor with conditional transformers
transformers = []

if none_fill:
    transformers.append(('none_cat', none_pipeline, none_fill))

if mode_fill:
    transformers.append(('mode_cat', mode_pipeline, mode_fill))

if num_fill:
    transformers.append(('num', num_pipeline, num_fill))

# Build preprocessor
preprocessor = ColumnTransformer(transformers, remainder='drop')

X_preprocessed = preprocessor.fit_transform(X)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=5)

#Build and train multi-regression model on training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict test data
y_pred = regressor.predict(X_test)



#Predict values of actual test dataset
test_df = pd.read_csv(r'data\test.csv')
X_test_actual = test_df.iloc[:, 1:].copy()
X_test_preprocessed = preprocessor.transform(X_test_actual)
y_test_pred = regressor.predict(X_test_preprocessed)

#Make csv file containing prediction
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': y_test_pred
})
submission.to_csv(r'data\submission.csv', index=False)