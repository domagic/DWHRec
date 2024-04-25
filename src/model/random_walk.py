import datetime
import gc
import os
import pickle
import time

import numpy as np
from numpy import random
from src.model.embedding_transform import walk2embedding


def get_all_edge_and_vertex_weight(para_hyperedges):
    edges_weights_dict = {}
    vertices_weights_dict = {}
    for edge in para_hyperedges:
        edges_weights_dict[edge] = para_hyperedges[edge]['e_weight']
        v_members = para_hyperedges[edge]['v_members']
        v_weights = para_hyperedges[edge]['v_weights']
        for member, weight in zip(v_members, v_weights):
            if member not in vertices_weights_dict:
                vertices_weights_dict[member] = weight
            else:
                vertices_weights_dict[member] += weight
        para_hyperedges[edge].pop('e_weight', None)
        para_hyperedges[edge].pop('v_weights', None)
    return edges_weights_dict, vertices_weights_dict


def choose_next_edge(para_vertex, para_vertex_links, para_edges_weights_dict):
    links = para_vertex_links[para_vertex]
    link_weights = np.vectorize(para_edges_weights_dict.get)(links)
    total_weights = link_weights.sum()
    if np.isinf(total_weights) or np.isnan(total_weights):
        raise ValueError("Total weights is inf or nan.")
    norm_weights = np.zeros_like(link_weights, dtype=float)
    for i in range(len(link_weights)):
        value = link_weights[i] / total_weights
        if np.isinf(value) or np.isnan(value):
            continue
        norm_weights[i] = value
    return links[random.choice(len(links), p=norm_weights)]


def update_edge_weight(para_edges_weights_dict, para_vertices_weights_dict, para_selected_edge, para_cur_vertex,
                       para_next_vertex):
    cur_vertex_weight = para_vertices_weights_dict[para_cur_vertex]
    next_vertex_weight = para_vertices_weights_dict[para_next_vertex]
    para_edges_weights_dict[para_selected_edge] += np.round(cur_vertex_weight * next_vertex_weight, 4)


def update_vertex_weight(para_edges_weights_dict, para_vertices_weights_dict, para_selected_edge, para_cur_vertex,
                         para_next_vertex):
    cur_vertex_weight = para_vertices_weights_dict[para_cur_vertex]
    next_vertex_weight = para_vertices_weights_dict[para_next_vertex]
    edge_weight = para_edges_weights_dict[para_selected_edge]
    para_vertices_weights_dict[para_cur_vertex] += np.round(cur_vertex_weight * edge_weight, 4)
    para_vertices_weights_dict[para_next_vertex] += np.round(next_vertex_weight * edge_weight, 4)


# def change_edge(para_cur_edge, para_cur_members, para_):
#     if random.randint(2):
#         adjacent_vertices = para_cur_members
#         candidate_weights = [vertices_weights[key] for key in adjacent_vertices]
#         candidate_weights = np.array(candidate_weights) / sum(candidate_weights)
#         hyperedge_adjacent = random.choice(adjacent_vertices, p=candidate_weights)
#         curr_hyperedge = para_hyperedges[hyperedge_adjacent]
#         next_nodes = list(curr_hyperedge["members"][:])
#     else:
#         return para_cur_edge

def choose_next_vertex(para_next_edge, para_cur_vertex, para_vertices_weights_dict):
    candidate_vertices = np.array(para_next_edge['v_members'])
    candidate_weights = np.vectorize(para_vertices_weights_dict.get)(candidate_vertices)
    condition = (candidate_vertices == para_cur_vertex)
    candidate_weights[condition] = 0

    total_weights = candidate_weights.sum()
    if np.isinf(total_weights) or np.isnan(total_weights):
        raise ValueError("Total weights is inf or nan.")
    norm_weights = np.zeros_like(candidate_weights, dtype=float)
    for i in range(len(candidate_weights)):
        value = candidate_weights[i] / total_weights
        if np.isinf(value) or np.isnan(value):
            continue
        norm_weights[i] = value

    selected_vertex = random.choice(candidate_vertices, p=norm_weights)
    return selected_vertex


# def normalize_edge_weight(para_edges_weights_dict):
#     weights = np.array(list(para_edges_weights_dict.values()))
#     sum_weights = np.sum(weights)
#     for edge in para_edges_weights_dict:
#         para_edges_weights_dict[edge] = para_edges_weights_dict[edge] / sum_weights


# def normalize_vertex_weight(para_edges_weights_dict, para_vertices_weights_dict, para_vertex_links):
#     for vertex in para_vertices_weights_dict:
#         cur_vertex_weight = np.array(para_vertices_weights_dict[vertex])
#         added_weight = cur_vertex_weight * np.vectorize(para_edges_weights_dict.get)(para_vertex_links[vertex])
#         para_vertices_weights_dict[vertex] = np.sum(added_weight)

def walk(para_hyperedges, para_vertex_links, para_num_iteration, para_num_steps):
    """
    Define a vertical random walk
    :param para_hyperedges:
    :param para_vertex_links:
    :param para_num_iteration:
    :param para_num_steps:
    :return:
    """
    walks_sat = []
    edges_weights_dict, vertices_weights_dict = get_all_edge_and_vertex_weight(para_hyperedges)
    np.random.default_rng()
    for cur_vertex in para_vertex_links:
        walk_path = []
        for turn in range(para_num_iteration):
            for step in range(0, para_num_steps + 1):
                walk_path.append(cur_vertex)
                next_edge = choose_next_edge(cur_vertex, para_vertex_links, edges_weights_dict)
                next_vertex = choose_next_vertex(para_hyperedges[next_edge], cur_vertex, vertices_weights_dict)
                # update_edge_weight(edges_weights_dict, vertices_weights_dict, next_edge, cur_vertex, next_vertex)
                # print(max(edges_weights_dict.values()), min(edges_weights_dict.values()))
                # update_vertex_weight(edges_weights_dict, vertices_weights_dict, next_edge, cur_vertex, next_vertex)
                # print(max(vertices_weights_dict.values()), min(vertices_weights_dict.values()))
                cur_vertex = next_vertex
            walks_sat.append(walk_path)
        # normalize_edge_weight(edges_weights_dict)
        # normalize_vertex_weight(edges_weights_dict, vertices_weights_dict, para_vertex_links)
    return walks_sat


def save_walks_to_file(para_walks_sat, para_result_folder):
    if not os.path.exists(para_result_folder):
        os.makedirs(para_result_folder)
    if len(os.listdir(para_result_folder)) != 0:
        for f in os.listdir(para_result_folder):
            os.remove(os.path.join(para_result_folder, f))

    delta = len(para_walks_sat) // 50
    for i in range(0, len(para_walks_sat), delta):
        if i + delta < len(para_walks_sat):
            filename = "walks_sat_" + str(i) + "-" + str(i + delta)
            pickle.dump(para_walks_sat[i:i + delta], open(os.path.join(para_result_folder, filename), "wb"))
        else:
            filename = "walks_sat_" + str(i) + "-" + str(len(para_walks_sat))
            pickle.dump(para_walks_sat[i:], open(os.path.join(para_result_folder, filename), "wb"))


# # connection feature
# def generate_embeddings(para_row, para_e_dict):
#     embedding = np.concatenate((context_embeddings[para_row[0]], context_embeddings[para_row[0]]))
#     para_e_dict[para_row[0]] = embedding


# # cos sim
# def cos_sim(a, b, w):
#     return 1 - spatial.distance.cosine(a, b, w)


def random_walks_embedding_phase(para_hyperedges, para_vertex_links, para_control_args):
    walk_result_folder = os.path.join(para_control_args['path']['temp_folder'],
                                      'walk', str(para_control_args['dataset']['fold']))
    if para_control_args['model']['walk']:
        start_time = datetime.datetime.now()
        print(
            ' '.join(['Fold:', str(para_control_args['dataset']['fold']), 'compute random walks at:', str(start_time)]))
        walks_sat = walk(para_num_iteration=para_control_args['parameter']['r'],
                         para_num_steps=para_control_args['parameter']['k'], para_hyperedges=para_hyperedges,
                         para_vertex_links=para_vertex_links)
        end_time = datetime.datetime.now()
        print(' '.join(
            ['Fold:', str(para_control_args['dataset']['fold']), 'finish random walks at:', str(end_time),
             ', cost time:',
             str((end_time - start_time).seconds / 3600), 'h.']))
        print(' '.join(['Fold:', str(para_control_args['dataset']['fold']), 'save walks result to file.']))
        save_walks_to_file(walks_sat, walk_result_folder)
        gc.collect()

    if para_control_args['model']['embedding']:
        walk2embedding(walk_result_folder, para_control_args)
