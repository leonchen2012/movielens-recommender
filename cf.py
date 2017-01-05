from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
from mltools import *


def rmse(pred, ground_truth):
    pred = pred[ground_truth.nonzero()].flatten()
    idx = np.where(np.isnan(pred))
    pred[idx] = 5
    idx = np.where(np.isinf(pred))
    pred[idx] = 5
    for i in range(len(pred)):
        if pred[i] > 5:
            pred[i] = 5
        if pred[i] < 1:
            pred[i] = 1
    pred = np.around(pred)
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, ground_truth))


def cf():
    data = loadml_1m()
    train_data, test_data = split_train_test(data)

    uid_list = data.userid.unique()
    movie_list = data.movieid.unique()
    num_user = uid_list.shape[0]
    num_movie = movie_list.shape[0]

    # There are movie ids that are larger than num_movie, have to re-index
    uid_mapper = {uid_list[i]: i for i in range(0, num_user)}
    movie_mapper = {movie_list[i]: i for i in range(0, num_movie)}

    print("There are %d users and %d movies\n" % (num_user, num_movie))

    train_data_matrix = np.zeros((num_user, num_movie))
    for line in train_data.itertuples():
        train_data_matrix[uid_mapper[line[1]], movie_mapper[line[2]]] = line[3]

    test_data_matrix = np.zeros((num_user, num_movie))
    for line in test_data.itertuples():
        test_data_matrix[uid_mapper[line[1]], movie_mapper[line[2]]] = line[3]

    print("start computing simularity")
    # user_sim = pairwise_distances(train_data_matrix, metric='cosine')
    item_sim = pairwise_distances(train_data_matrix.T, metric='cosine')

    print("start predicting")
    # based on item simularity
    mask = np.copy(train_data_matrix)
    mask[train_data_matrix > 0] = 1
    item_pred = train_data_matrix.dot(item_sim) / mask.dot(np.abs(item_sim))

    # based on user simularity
    # do a normalization trick
    # mean_user_rating = train_data_matrix.mean(axis=1)
    # rating_diff = train_data_matrix - mean_user_rating[:, np.newaxis]
    # user_pred = mean_user_rating[:, np.newaxis] + user_sim.dot(rating_diff)/np.array([np.abs(user_sim).sum(axis=1)]).T

    # print("User based rmse %f\n" % rmse(user_pred, test_data_matrix))
    print("Item based rmse %f\n" % rmse(item_pred, test_data_matrix))

if __name__ == "__main__":
    cf()