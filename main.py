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


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.head(5))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

# Save training 'sale price'
train_SalePrice = train['SalePrice']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
print("The test data size after dropping Id feature is : {} ".format(test.shape))

# Plan of attack:
# Tue+ Wed: Dataset Cleanup: Find and drop outliers, input missing data into fields, drop unnessesary features
# - Figure out which features correlate best with sales price
# - Figure out which features have a lot of missing values
# - Drop unnessary features
# - Find and drop major outliers
# - Input missing values
# Thursday: Apply models (KNN, Ridge, Neural Networks), First Draft of Report
# Friday: Finish Report

# START HERE



#

# fig, ax = plt.subplots()
# ax.scatter(x = train['GrLivArea'], y = train_SalePrice)
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()

# Delete outliers
# train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


# fig, ax = plt.subplots()
# ax.scatter(x = train['GrLivArea'], y = train_SalePrice)
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)




# sns.distplot(train_SalePrice , fit=norm);
# Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(train['SalePrice'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')


#####
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
# train["SalePrice"] = np.log1p(train_SalePrice)

#Check the new distribution
# sns.distplot(train_SalePrice , fit=norm);

# Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(train["SalePrice"])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
# plt.show()

# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()


# Features Engineering
# train.drop("SalePrice", axis = 1, inplace = True)

# ntrain = train.shape[0]
# ntest = test.shape[0]
# y_train = train.SalePrice.values
# all_data = pd.concat((train, test)).reset_index(drop=True)
# all_data.drop(['SalePrice'], axis=1, inplace=True)
# print("all_data size is : {}".format(all_data.shape))
#
# all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
# all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
# missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
# print(missing_data)
