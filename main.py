import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# Helper function to see the output of predictions
def print_predictions(regressor):
    for i in range(len(X_test)):
        print(f"{regressor.predict([X_test[i]])[0]}\t{y_test[i]}")

# Calculate the mean absolute error
def calcMeanABSError(regressor, xs, ys):
    total = 0
    for i in range(len(xs)):
        prediction = regressor.predict([xs[i]])[0]
        actual = ys[i]
        total += abs(prediction - actual)
    return total / len(xs)


def main():
    df = pd.read_csv('./AlgerianFF_Region1.csv')
    df.merge( pd.read_csv('./AlgerianFF_Region2.csv') )
    df.head()

    df = df.sample(frac=1).reset_index(drop=True)
    df.head()

    y = np.array(df['FWI'])
    # Drop target (FWI) from training data
    # Drop day and year as they won't help
    # Month is left in as the month could be a predictor of a forest fire
    df = df.drop(['FWI', 'day', 'year'], axis=1) 
    inputs = np.array(df, float)
    X = preprocessing.scale(inputs)

    split_point = math.floor(len(X) * 0.7)

    X_train, X_test = X[split_point:], X[split_point + 1: ]
    y_train, y_test = y[split_point:], y[split_point + 1: ]



        



    RFregr = RandomForestRegressor(max_depth=2).fit(X_train, y_train)
    RFregr.score(X_test, y_test)

    print("-------------RF-------------")


    print(calcMeanABSError(RFregr, X_test, y_test))


    print(calcMeanABSError(RFregr, X_train, y_train))


    MLPRegr = MLPRegressor(max_iter=1000).fit(X_train, y_train)
    MLPRegr.score(X_test, y_test)

    print("-------------MLP-------------")


    print(calcMeanABSError(MLPRegr, X_test, y_test))


    print(calcMeanABSError(MLPRegr, X_train, y_train))


    SVMRegr = SVR().fit(X_train, y_train)
    SVMRegr.score(X_test, y_test)

    print("-------------SVM-------------")
    print(calcMeanABSError(SVMRegr, X_test, y_test))


    print(calcMeanABSError(SVMRegr, X_train, y_train))




if __name__ == "__main__":
    main()