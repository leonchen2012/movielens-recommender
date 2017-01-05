from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
from mltools import *


def pyfm_pred():
    data = loadml_1m()
    train_data, test_data = split_train_test(data, True)
    print("start vectorizer")
    v = DictVectorizer()
    X_train = v.fit_transform(train_data[['userid', 'movieid']].to_dict('records'))
    X_test = v.fit_transform(test_data[['userid', 'movieid']].to_dict('records'))
    fm = pylibfm.FM(num_factors=10, num_iter=20, verbose=True, task="regression", initial_learning_rate=0.001,
                    learning_rate_schedule="optimal")

    print("start training")
    fm.fit(X_train, train_data.rating)
    print("start prediction")
    preds = fm.predict(X_test)

    print("FM MSE: %.4f" % mean_squared_error(test_data.rating, preds))


if __name__ == "__main__":
    pyfm_pred()
