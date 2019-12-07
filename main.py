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
from sklearn.preprocessing import StandardScaler

def main():

    # Read the training data
    train = pd.read_csv('train.csv')

    # Drop the  'Id' colum since it's unnecessary for prediction
    train.drop("Id", axis = 1, inplace = True)

    # Show the correlation matrix, plot grlivarea vs sale price
    corr_matrix(train)

    # Remove Outliers
    train = remove_outliers(train)
    # Show data with outliers removed
    data = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
    data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))


    # Check how much data is missing
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']);
    print(f"\n{missing_data.head(20)}")

    # Delete rows/columns with missing data
    train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
    train = train.drop(train.loc[train['Electrical'].isnull()].index)

    # Check how much data is missing (agian)
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']);
    print(f"\nMissing Data: {train.isnull().sum().max()}")

    # These variables should be categorical not numeric
    train['MSSubClass'] = train['MSSubClass'].apply(str)
    train['OverallCond'] = train['OverallCond'].astype(str)
    train['YrSold'] = train['YrSold'].astype(str)
    train['MoSold'] = train['MoSold'].astype(str)

    X_vals = train.drop(['SalePrice'], axis = 1)
    # Normalization and convert categorical variables to dummies
    X_vals = normalize_dummy(X_vals)
    X_vals = X_vals.values
    Y_vals = train['SalePrice'].values

    # Test-Train Split
    from sklearn.model_selection import train_test_split
    (x_train, x_test, y_train, y_test) = train_test_split(X_vals, Y_vals, test_size = 0.2)

    # Normalize y_train values
    y_train = np.log1p(y_train)
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train.reshape(-1,1))
    y_train.reshape(-1)

    # Import ML Algorithms
    from sklearn.model_selection import GridSearchCV, cross_validate
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.neighbors import KNeighborsRegressor

    # Tuning Hyperparameters
    import time

    # KRR
    # For the figure in the report (I took the scores and used excel to make the graph)
    alphas = [0.001, 0.01, 0.1, 1, 10]
    score_poly2 = []
    score_poly3 = []
    score_rbf = []
    for alpha in alphas:
        m1 = KernelRidge(alpha=alpha, kernel='polynomial', degree=2)
        m2 = KernelRidge(alpha=alpha, kernel='polynomial', degree=3)
        m3 = KernelRidge(alpha=alpha, kernel='rbf')
        cv_1 = cross_validate(m1, x_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        cv_2 = cross_validate(m2, x_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        cv_3 = cross_validate(m3, x_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        score_poly2.append(np.negative(np.mean(cv_1['test_score'])))
        score_poly3.append(np.negative(np.mean(cv_2['test_score'])))
        score_rbf.append(np.negative(np.mean(cv_3['test_score'])))
    print(f"\nAverage Cross Validation Scores for Alpha = [0.001, 0.01, 0.1, 1, 10], Various Kernels\n2nd Degree Polynomial: {score_poly2}\n3rd Degree Polynomial: {score_poly3}\nRBF Kernel: {score_rbf}")

    # Use grid Search to find best parameters
    t_krr_start = time.time()

    parameters = {
        'alpha' : [0.8, 0.9, 1, 1.1, 1.2],
        }
    grid_search_krr = GridSearchCV(KernelRidge(), parameters, cv=5)
    grid_search_krr.fit(x_train, y_train)
    best_params_krr = grid_search_krr.best_params_
    KRR = KernelRidge(alpha=best_params_krr['alpha'], kernel='polynomial')

    t_krr_elapsed = time.time()-t_krr_start
    print(f"\nElapsed Time for KRR Grid Search: {t_krr_elapsed}")
    print(f"\nBest alpha parameter found: {best_params_krr}")

    # KNR
    # For the figure in the report (I took the scores and used excel to make the graph)
    neighbors = np.arange(1,21)
    score_euclidean = []
    score_manhattan = []
    for n in neighbors:
        m1 = KNeighborsRegressor(n_neighbors=n, metric='euclidean')
        m2 = KNeighborsRegressor(n_neighbors=n, metric='manhattan')
        cv_1 = cross_validate(m1, x_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        cv_2 = cross_validate(m2, x_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        score_euclidean.append(np.negative(np.mean(cv_1['test_score'])))
        score_manhattan.append(np.negative(np.mean(cv_2['test_score'])))
    print(f"\nAverage Cross Validation Scores for K in range [1-20], Euc and Man Distance Metrics\nEuclidian Metric: {score_euclidean}\nManhattan Metric: {score_manhattan}")

    # Grid Search CV KNR
    t_knr_start = time.time()
    parameters = {'n_neighbors' : np.arange(1,16)}
    grid_search_knr = GridSearchCV(KNeighborsRegressor(), parameters, cv=5)
    grid_search_knr.fit(x_train, y_train)
    best_params_knr = grid_search_knr.best_params_
    KNR = KNeighborsRegressor(n_neighbors=best_params_knr['n_neighbors'], metric='manhattan')
    t_knr_elapsed = time.time()-t_knr_start
    print(f"\nElapsed Time KNR Grid Search: {t_knr_elapsed}")
    print(f"\nBest k-value found (manhattan distance metric): {best_params_knr}")

    # Build and Evaluate Neural Networks
    # I tested each of them out (Keras outputs rmse loss and elapsed training time)
    from keras.models import Sequential
    from keras.layers import Dense

    # 128 Inner Dimensions
    print("\nBuilding Multiple Neural Networks")
    # NN_model = Sequential()
    # NN_model.add(Dense(128, kernel_initializer='normal',input_dim = x_train.shape[1], activation='relu'))
    # NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
    # NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
    # NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    # NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    # NN_model.fit(x_train, y_train)

    # 256 Inner dimensions
    NN_model = Sequential()
    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = x_train.shape[1], activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    NN_model.fit(x_train, y_train)


    # 512 Inner dimensions
    # NN_model = Sequential()
    # NN_model.add(Dense(128, kernel_initializer='normal',input_dim = x_train.shape[1], activation='relu'))
    # NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
    # NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
    # NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    # NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    # NN_model.fit(x_train, y_train)
    #
    # # 1024
    # NN_model = Sequential()
    # NN_model.add(Dense(128, kernel_initializer='normal',input_dim = x_train.shape[1], activation='relu'))
    # NN_model.add(Dense(1024, kernel_initializer='normal',activation='relu'))
    # NN_model.add(Dense(1024, kernel_initializer='normal',activation='relu'))
    # NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    # NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    # NN_model.fit(x_train, y_train)


    # Fit and predict for each
    KRR.fit(x_train, y_train)
    krr_predict = KRR.predict(x_test)
    KNR.fit(x_train, y_train)
    knr_predict = KNR.predict(x_test)
    NN_model.fit(x_train, y_train)
    nnm_predict = NN_model.predict(x_test)
    nnm_predict = nnm_predict.reshape(-1)

    # Rescale predicted values
    krr_predict = rescale(krr_predict, scaler)
    knr_predict = rescale(knr_predict, scaler)
    nnm_predict = rescale(nnm_predict, scaler)

    # Get MAE and RMSE Values for each
    from sklearn.metrics import mean_squared_error, r2_score

    krr_rmse = mean_squared_error(y_test, krr_predict, squared=False)
    knr_rmse = mean_squared_error(y_test, knr_predict, squared=False)
    nnm_rmse = mean_squared_error(y_test, nnm_predict, squared=False)
    krr_r2 = r2_score(y_test, krr_predict)
    knr_r2 = r2_score(y_test, knr_predict)
    nnm_r2 = r2_score(y_test, nnm_predict)

    # Evaluate Algorithms
    print(f"\nRidge Regression {krr_rmse, krr_r2}\nNeighbors Regression: {knr_rmse, knr_r2}\nNeural Network {nnm_rmse, nnm_r2}")
    print(f"\nMean House Sale Price {np.mean(y_test)}")
    plt.show()





def corr_matrix(df_train):
    # Correlation Matrix
    k = 5
    corrmat = df_train.corr()
    f, ax = plt.subplots(figsize=(8, 8))
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale=1.0)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    # GrLivArea with outliers
    var = 'GrLivArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))


def remove_outliers(train):
    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
    return train

def rescale(y_vals, scaler):
    y_vals = scaler.inverse_transform(y_vals)
    y_vals = np.expm1(y_vals)
    return y_vals

def normalize_dummy(train):

    # Normalize numerical values
    numeric_feats = train.dtypes[train.dtypes != "object"].index
    for feat in numeric_feats:
        train[feat] = np.log1p(train[feat])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(train[numeric_feats].values)
    train[numeric_feats] = scaled_features

    # Dummy Variables
    train = pd.get_dummies(train)
    # Check to see if variables are normalized
    # sns.distplot(train['1stFlrSF'] , fit=norm)
    # print(train.head())
    return train

if __name__ == "__main__":
    main()
