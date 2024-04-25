import datetime
import os
import pickle
import time

import numpy as np
from gensim.models import Word2Vec


def load_walks_from_file(para_file_dir):
    res_walks_sat = []
    for f in os.listdir(para_file_dir):
        piked = open(os.path.join(para_file_dir, f), 'rb')
        for el in pickle.load(piked):
            res_walks_sat.append(el)
        piked.close()
    return res_walks_sat


# Generate context embeddings
def EmbedWord2Vec(para_walks, para_dimension, para_window_size):
    model = Word2Vec(para_walks, vector_size=para_dimension, window=para_window_size, min_count=0, sg=1, workers=20,
                     epochs=20)
    node_ids = model.wv.index_to_key
    node_embeddings = model.wv.vectors
    return node_ids, node_embeddings


def walk2embedding(para_walk_result_folder, para_control_args):
    fold = para_control_args['dataset']['fold']
    print(' '.join(['Fold:', str(fold), 'loading walks data from file.']))
    walks_sat = load_walks_from_file(para_walk_result_folder)

    vertex_embedding_dimension = para_control_args['parameter']['s']
    # creation of node embeddings
    start_time = datetime.datetime.now()
    print(' '.join(['Fold:', str(fold), 'creation of vertex embeddings at:', str(start_time)]))
    vertex_ids, vertex_embeddings = EmbedWord2Vec(para_walks=walks_sat, para_dimension=vertex_embedding_dimension,
                                                  para_window_size=para_control_args['parameter']['w'])
    end_time = datetime.datetime.now()
    print(' '.join(
        ['Fold:', str(fold), "vertex embeddings complete at:", str(end_time), '(' + str(len(vertex_embeddings)),
         ' embeddings)', 'cost time:', str((end_time - start_time).seconds / 3600), 'h.']))

    vertex_ids_file_path = os.path.join(para_control_args['path']['temp_folder'], 'embedding',
                                        ''.join(['vertex_ids_fold_', str(fold), '.npy']))
    if not os.path.exists(os.path.dirname(vertex_ids_file_path)):
        os.makedirs(os.path.dirname(vertex_ids_file_path))
    vertex_embeddings_file_path = os.path.join(os.path.dirname(vertex_ids_file_path),
                                               ''.join(['vertex_embeddings_fold_', str(fold), '.npy']))
    np.save(vertex_ids_file_path, vertex_ids)
    np.save(vertex_embeddings_file_path, vertex_embeddings)
