# tool functions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def loadml_1m():
    header = ['userid', 'movieid', 'rating', 'rating_time']
    rating_list = pd.read_csv('ml-1m/ratings.dat', sep='::', names=header, engine='python')
    return rating_list


def split_train_test(data, changetype=False):
    if changetype:
        data.userid = data.userid.astype(np.str)
        data.movieid = data.movieid.astype(np.str)
        data.rating = data.rating.astype(np.float)
    return train_test_split(data, test_size=0.25, random_state=0)

