# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# import seaborn as sns
# import statsmodels.formula.api as smf
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# import warnings
#
# warnings.filterwarnings('ignore')
#
# plt.style.use('ggplot')
#
# data = load_boston()  # reading data
#
# # creating dataframe
# df = pd.DataFrame(data.data, columns=data.feature_names)
#
# # adding target value to the data
# df['MEDV'] = data.target
#
# print('Details of the Database\n')
# df.head()
# df.describe()
# df.info()
#
# # looking at null values
# df.isna().sum()
#
# # Let's see how data is distributed for every column using data Visualization
# plt.figure(figsize=(20, 15))
# plotnumber = 1
#
# for column in df:
#     if plotnumber <= 14:
#         ax = plt.subplot(3, 5, plotnumber)
#         sns.distplot(df[column])
#         plt.xlabel(column, fontsize=15)
#
#     plotnumber += 1
#
# plt.tight_layout()
# plt.show()
#
# # Plotting `Price` with remaining columns
# plt.figure(figsize=(20, 15))
# plotnumber = 1
#
# for column in df:
#     if plotnumber <= 14:
#         ax = plt.subplot(3, 5, plotnumber)
#         sns.scatterplot(x=df['MEDV'], y=df[column])
#
#     plotnumber += 1
#
# plt.tight_layout()
# plt.show()
#
# # looking for outliers using box plot
# plt.figure(figsize=(20, 8))
# sns.boxplot(data=df, width=0.8)
# plt.show()
#
# # creating features and label variable
# X = df.drop(columns='MEDV', axis=1)
# y = df['MEDV']
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled
#
# # checking for multicollinearity using `VIF` and `correlation matrix`
#
# vif = pd.DataFrame()
#
# vif['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
# vif['Features'] = X.columns
#
# vif
#
# # Heatmap
# fig, ax = plt.subplots(figsize=(16, 8))
# sns.heatmap(df.corr(), annot=True, fmt='1.2f', annot_kws={'size': 10}, linewidth=1)
# plt.show()
#
# lm = smf.ols(formula='MEDV ~ RAD', data=df).fit()
# lm.summary()
#
# # Notes:[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# lm = smf.ols(formula='MEDV ~ TAX', data=df).fit()
# lm.summary()
#
# # From OLS Regression Results we can conclude that removing "RAD" column will be good.
# # removing "RAD" column
#
# df.drop(columns='RAD', axis=1, inplace=True)
# df.head()
#
# # splitting data into training asnd test set
#
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=0)
# # fitting training data to model
#
# lr = LinearRegression()
# lr.fit(X_train, y_train)
#
# # prediction of model
# y_pred = lr.predict(X_test)
#
# # training accuracy of model
# print('Training Accuracy of the model = ', lr.score(X_train, y_train))
#
# # test accuracy of model
# print('Testing Accuracy of the model = ', lr.score(X_test, y_test))
#
#
# # creating a function to create adhusted R-Squared
# def adj_r2(X, y, model):
#     r2 = model.score(X, y)
#     n = X.shape[0]
#     p = X.shape[1]
#     adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
#
#     return adjusted_r2
#
#
# print('\nAfter adjusted R-squared New values of Accuracy\n')
#
# print('Training Accuracy of the model = ', adj_r2(X_train, y_train, lr))
# print('Testing Accuracy of the model = ', adj_r2(X_test, y_test, lr))
#
# # Model r2 score is less on the test data so there is chance of overfitting,


from matplotlib import pyplot as plt
import numpy as np
import random
import utils

features = np.array([1,2,3,5,6,7])
labels = np.array([155, 197, 244, 356,407,448])

print(features)
print(labels)

utils.plot_points(features, labels)

def simple_trick(base_price, price_per_room, num_rooms, price):
    small_random_1 = random.random()*0.1
    small_random_2 = random.random()*0.1
    predicted_price = base_price + price_per_room*num_rooms
    if price > predicted_price and num_rooms > 0:
        price_per_room += small_random_1
        base_price += small_random_2
    if price > predicted_price and num_rooms < 0:
        price_per_room -= small_random_1
        base_price += small_random_2
    if price < predicted_price and num_rooms > 0:
        price_per_room -= small_random_1
        base_price -= small_random_2
    if price < predicted_price and num_rooms < 0:
        price_per_room -= small_random_1
        base_price += small_random_2
    return price_per_room, base_price

def absolute_trick(base_price, price_per_room, num_rooms, price, learning_rate):
    predicted_price = base_price + price_per_room*num_rooms
    if price > predicted_price:
        price_per_room += learning_rate*num_rooms
        base_price += learning_rate
    else:
        price_per_room -= learning_rate*num_rooms
        base_price -= learning_rate
    return price_per_room, base_price

def square_trick(base_price, price_per_room, num_rooms, price, learning_rate):
    predicted_price = base_price + price_per_room*num_rooms
    price_per_room += learning_rate*num_rooms*(price-predicted_price)
    base_price += learning_rate*(price-predicted_price)
    return price_per_room, base_price

import random as rd

# We set the random seed in order to always get the same results.
random.seed(0)

def linear_regression(features, labels, learning_rate=0.01, epochs = 1000):
    price_per_room = random.random()
    base_price = random.random()
    for epoch in range(epochs):
        # Uncomment any of the following lines to plot different epochs
        #if epoch == 1:
        #if epoch <= 10:
        #if epoch <= 50:
        #if epoch > 50:
        if True:
            utils.draw_line(price_per_room, base_price, starting=0, ending=8)
        i = random.randint(0, len(features)-1)
        num_rooms = features[i]
        price = labels[i]
        # Uncomment any of the 2 following lines to use a different trick
        #price_per_room, base_price = absolute_trick(base_price,
        price_per_room, base_price = square_trick(base_price,
                                                  price_per_room,
                                                  num_rooms,
                                                  price,
                                                  learning_rate=learning_rate)
    utils.draw_line(price_per_room, base_price, 'black', starting=0, ending=8)
    utils.plot_points(features, labels)
    print('Price per room:', price_per_room)
    print('Base price:', base_price)
    return price_per_room, base_price

# This line is for the x-axis to appear in the figure
plt.ylim(0,500)

linear_regression(features, labels, learning_rate = 0.01, epochs = 1000)


def rmse(labels, predictions):
    n = len(labels)
    differences = np.subtract(labels, predictions)
    return np.sqrt(1.0/n * (np.dot(differences, differences)))

def linear_regression(features, labels, learning_rate=0.01, epochs = 1000):
    price_per_room = random.random()
    base_price = random.random()
    errors = []
    for i in range(epochs):
        predictions = features[0]*price_per_room+base_price
        errors.append(rmse(labels, predictions))
        i = random.randint(0, len(features)-1)
        num_rooms = features[i]
        price = labels[i]
        # Uncomment one of the following 3 lines to use the simple, the absolute, or the square trick
        #price_per_room, base_price = simple_trick(base_price,
        #price_per_room, base_price = absolute_trick(base_price,
        price_per_room, base_price = square_trick(base_price,
                                                  price_per_room,
                                                  num_rooms,
                                                  price,
                                                  learning_rate=learning_rate)
    utils.draw_line(price_per_room, base_price, 'black', starting=0, ending=9)
    utils.plot_points(features, labels)
    print('Price per room:', price_per_room)
    print('Base price:', base_price)
    plt.show()
    plt.scatter(range(len(errors)), errors)
    plt.show()
    return price_per_room, base_price

linear_regression(features, labels, learning_rate = 0.01, epochs = 10000)

import turicreate as tc

datadict = {
    'features': features,
    'labels'  : labels,
}

data = tc.SFrame(datadict)
