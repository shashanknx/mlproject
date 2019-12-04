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
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import StandardScaler
# Measure Elapsed Time
import time 


# Data Cleanup:
# Target Variable Distribution
# Remove Missing Variables
# Remove Outliers
# Remove Correlated Variables
# Convert some numerical values into categorical
# StandardScaler all features
# Do Heat Transfer
# Modelling:
# Test-Train Split
# Tune Hyperparameters
# Run on Test Data
# Calculate RSME Values between predicted and Test Data
# Finish off report

# Read the training data
train = pd.read_csv('train.csv')
print(train.head(5))

#Save the 'Id' column
train_ID = train['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)

#Check the size of the data
print(f"The train data size after dropping Id feature is : {train.shape}")

#

k = 10
corrmat = train.corr()
f, ax = plt.subplots(figsize=(8,8))
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']
sns.pairplot(train[cols], height=2)

# plt.show()

#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(21))

print(train.shape)

#dealing with missing data
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
print(train.isnull().sum().max()) #just checking that there's no missing data missing...


#standardizing data
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

fig, ax = plt.subplots()
print(train['GrLivArea'].shape, train['SalePrice'].shape)
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
# Delete outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

sns.distplot(train['SalePrice'] , fit=norm);
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# Save training 'sale price'
train_SalePrice = train['SalePrice']

# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train['SalePrice'] = np.log1p(train_SalePrice)

# Check the new distribution
sns.distplot(train['SalePrice'], fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train["SalePrice"])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

train = pd.get_dummies(train)
