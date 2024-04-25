import os

import numpy as np
import pandas as pd
from scipy.stats import zscore


# compute scores and top-k prediction for each user
def compute_score_matrix(para_user_embeddings_array, para_track_embeddings_array):
    num = np.dot(para_user_embeddings_array, para_track_embeddings_array.T)
    return num


def recommendations_phase(para_embeddings_folder, para_control_args):
    if not para_control_args['model']['recommendation']:
        return
    cur_fold = para_control_args['dataset']['fold']
    print(' '.join(['Fold:', str(cur_fold), 'recommendation.']))
    vertex_ids_file_path = os.path.join(para_embeddings_folder,
                                        ''.join(['vertex_ids_fold_', str(cur_fold), '.npy']))
    vertex_embeddings_file_path = os.path.join(para_embeddings_folder,
                                               ''.join(['vertex_embeddings_fold_', str(cur_fold), '.npy']))
    vertex_ids = np.load(vertex_ids_file_path)
    vertex_embeddings = np.load(vertex_embeddings_file_path)

    # create embedding users
    user_embeddings_dict = {}
    # create embedding tracks
    track_embeddings_dict = {}
    for v_id, v_vector in zip(vertex_ids, vertex_embeddings):
        if v_id.startswith('u_'):
            user_embeddings_dict[v_id] = v_vector
        elif v_id.startswith('tr_'):
            track_embeddings_dict[v_id] = v_vector
    user_embeddings_df = pd.DataFrame(data=np.zeros(shape=(len(user_embeddings_dict), 3)), columns=['id', 'uid', 'v'])
    user_embeddings_df['id'] = np.arange(0, len(user_embeddings_dict))
    user_embeddings_df['uid'] = 'u_' + user_embeddings_df['id'].astype(str)
    user_embeddings_df['v'] = user_embeddings_df['uid'].map(user_embeddings_dict)

    track_embeddings_df = pd.DataFrame(data=np.zeros(shape=(len(track_embeddings_dict), 3)), columns=['id', 'iid', 'v'])
    track_embeddings_df['id'] = np.arange(0, len(track_embeddings_dict))
    track_embeddings_df['iid'] = 'tr_' + track_embeddings_df['id'].astype(str)
    track_embeddings_df['v'] = track_embeddings_df['iid'].map(track_embeddings_dict)

    # check nan
    for i in range(len(track_embeddings_df)):
        v_vector = track_embeddings_df.loc[i, 'v']
        if np.any(np.isnan(v_vector)):
            arr = [track_embeddings_df.loc[i, 'id'], track_embeddings_df.loc[i, 'iid'],
                   np.zeros(shape=(para_control_args['parameter']['s'],))]
            track_embeddings_df.iloc[i] = pd.Series(arr)

    # user embeddings to array
    user_embeddings_array = np.vstack(user_embeddings_df['v'].to_numpy())
    track_embeddings_array = np.vstack(track_embeddings_df['v'].to_numpy())

    # Normalization of embeddings
    user_embeddings_array = zscore(user_embeddings_array, ddof=1)
    track_embeddings_array = zscore(track_embeddings_array, ddof=1)

    # Recommendation system (weighted cosine distance)
    top_n = para_control_args['parameter']['n']
    users_predictions = np.zeros((len(user_embeddings_array), top_n), dtype=int)
    parameter = 1 - 1 / (np.arange(len(track_embeddings_array)) + 2)
    for t in range(len(user_embeddings_array)):
        users_score_matrix = compute_score_matrix(user_embeddings_array[t], track_embeddings_array)
        diversity_goal = 1 - users_score_matrix
        np.random.shuffle(parameter)
        users_score_matrix_with_parameter = parameter * diversity_goal
        # calculate top k suggestions
        ordered_candidate_tracks = np.argsort(users_score_matrix_with_parameter)[0:top_n]
        users_predictions[t] = ordered_candidate_tracks

    predictions_file_path = os.path.join(para_control_args['path']['result_folder'],
                                         ''.join(['predictions_fold_', str(cur_fold), '.csv']))
    if not os.path.exists(os.path.dirname(predictions_file_path)):
        os.makedirs(os.path.dirname(predictions_file_path))
    np.savetxt(predictions_file_path, users_predictions, fmt='%d', delimiter=',')
