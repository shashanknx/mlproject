# Linear Algebra
import numpy as np
# Data Processing
import pandas as pd
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
# Statistics
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler


# Read the training data
train = pd.read_csv('train.csv')
# print(train.head(5))

# Save the 'Id' column
train_ID = train['Id']

# Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)

# Check the size of the data
# print(f"The train data size after dropping Id feature is : {train.shape}")

# Outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
# data = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
# data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))

# Check how much data is missing
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']);
# print(missing_data.head(20))

# Delete rows/columns with missing data
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)

# Check how much data is missing (agian)
# total = train.isnull().sum().sort_values(ascending=False)
# percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']);
# print(f"\n{train.isnull().sum().max()}")

# These variables should not be numeric
train['MSSubClass'] = train['MSSubClass'].apply(str)
train['OverallCond'] = train['OverallCond'].astype(str)
train['YrSold'] = train['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)

# Normalize numerical values
numeric_feats = train.dtypes[train.dtypes != "object"].index
for feat in numeric_feats:
    train[feat] = np.log1p(train[feat])

scaled_features = StandardScaler().fit_transform(train[numeric_feats].values)
train[numeric_feats] = scaled_features

# Dummy Variables
train = pd.get_dummies(train)
# Check to see if variables are normalized
# sns.distplot(train['1stFlrSF'] , fit=norm)
# print(train.head())
plt.show()

# Import ML Algorithms
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor


# Test-Train Split
X_vals = train.drop(['SalePrice'], axis = 1)
X_vals = X_vals.values
Y_vals = train['SalePrice'].values
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X_vals, Y_vals, test_size = 0.2)

# Grid search CV KRR
parameters = {
    'alpha' : [0.1, .01, 1],
    'kernel' : ['polynomial', 'rbf']
    }
grid_search_krr = GridSearchCV(KernelRidge(), parameters, cv=5)
grid_search_krr.fit(x_train, y_train)
best_params_krr = grid_search_krr.best_params_
KRR = KernelRidge(alpha=best_params_krr['alpha'], kernel=best_params_krr['kernel'])

# Grid Search CV KNN
parameters = {'n_neighbors' : np.arange(1,16)}
grid_search_knr = GridSearchCV(KNeighborsRegressor(), parameters, cv=5)
grid_search_knr.fit(x_train, y_train)
best_params_knr = grid_search_knr.best_params_
KNR = KNeighborsRegressor(n_neighbors=best_params_knr['n_neighbors'])


# Build Neural Network
from keras.models import Sequential
from keras.layers import Dense
NN_model = Sequential()
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = x_train.shape[1], activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

# Fit and predict for each
KRR.fit(x_train, y_train)
krr_predict = KRR.predict(x_test)
KNR.fit(x_train, y_train)
knr_predict = KNR.predict(x_test)
NN_model.fit(x_train, y_train)
nnm_predict = NN_model.predict(x_test)
nnm_predict = nnm_predict.reshape(-1)
breakpoint()

# Get MAE and RMSE Values for each
from sklearn.metrics import mean_squared_error, r2_score

krr_rmse = mean_squared_error(y_test, krr_predict, squared=False)
knr_rmse = mean_squared_error(y_test, knr_predict, squared=False)
nnm_rmse = mean_squared_error(y_test, nnm_predict, squared=False)
krr_r2 = r2_score(y_test, krr_predict)
knr_r2 = r2_score(y_test, knr_predict)
nnm_r2 = r2_score(y_test, nnm_predict)

# Evaluate Algorithms
print(f"Ridge Regression {krr_rmse, krr_r2}\nNeighbors Regression: {knr_rmse, knr_r2}\nNeural Network {nnm_rmse, nnm_r2}")
