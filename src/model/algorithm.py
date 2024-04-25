import os
import pandas as pd

from src.model.hypergraph import hyperedges_construction_phase
from src.model.random_walk import random_walks_embedding_phase
from src.model.recommendation import recommendations_phase


def train_test_split(para_user_data):
    """
    Splitting the dataset into training and test sets.
    :param para_user_data: The inter dataset.
    :return: Training and test sets.
    """
    training = para_user_data[para_user_data['set'] == 'train'].copy().reset_index(drop=True)
    test = para_user_data[para_user_data['set'] == 'test'].copy().reset_index(drop=True)
    return training, test


def load_data(para_data_folder, para_fold):
    # Load data from a (CSV) file with a specific format to the current program.
    file_user_data = os.path.join(para_data_folder, 'inters_data_fold_' + str(para_fold) + '.csv')
    file_track_data = os.path.join(para_data_folder, 'tracks_data.csv')
    file_artist_data = os.path.join(para_data_folder, 'artists_data.csv')
    file_album_data = os.path.join(para_data_folder, 'albums_data.csv')
    inter_data = pd.read_csv(file_user_data)
    track_data = pd.read_csv(file_track_data)
    artist_data = pd.read_csv(file_artist_data)
    album_data = pd.read_csv(file_album_data)
    return inter_data, track_data, artist_data, album_data


def algorithm_body(para_control_args):
    fold = para_control_args['dataset']['fold']
    inter_data, track_data, artist_data, album_data = load_data(para_control_args['path']['data_folder'], fold)

    # train and test set split
    inter_training, inter_test = train_test_split(inter_data)
    hyperedges, vertex_links = hyperedges_construction_phase(inter_training, track_data, album_data, artist_data,
                                                             para_control_args)
    random_walks_embedding_phase(hyperedges, vertex_links, para_control_args)
    recommendations_phase(os.path.join(para_control_args['path']['temp_folder'], 'embedding'), para_control_args)
