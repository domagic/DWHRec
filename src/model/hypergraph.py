import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Vertex:
    def __init__(self, vid):
        self.vid = vid
        self.weight = 0
        self.links = None

    def set_weight(self, weight):
        self.weight = weight

    def set_connect(self, links):
        self.links = links

    def get_weight(self):
        return self.weight


class Hyperedge:
    def __init__(self, genre=None, head=None, tail=None, weight=None):
        self.type = genre
        self.head = head
        self.tail = tail
        self.weight = weight
        self.hid = None
        self.vertices = None
        self.vertex_weights = None

    def get_type(self):
        return self.type

    def get_head(self):
        return self.head

    def get_tail(self):
        return self.tail

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight

    def update_weight(self, weight):
        self.weight += weight


class Hypergraph:
    def __init__(self, vertices=np.empty(shape=0), hyperedges=np.empty(shape=0)):
        self.vertices = vertices
        self.hyperedges = hyperedges

    def info(self):
        vertices_info = np.array2string(self.vertices, separator=', ', formatter={'int': lambda x: f'{x:02}'})
        hyperedges_info = np.array2string(self.hyperedges, separator=', ', formatter={'int': lambda x: f'{x:02}'})
        return '\n'.join([vertices_info, hyperedges_info])


def user_tracks_hyperedges_construction(para_inter_data):
    res_user_hyperedges = para_inter_data.groupby('user_id_matrix')['track_id'].apply(list).reset_index(name='tracks')
    unique_user_id = para_inter_data[['user_id', 'user_id_matrix']].copy().drop_duplicates()
    user_play_count = para_inter_data.groupby('user_id_matrix')['play'].apply(list).reset_index(name='play_count')
    res_user_hyperedges = pd.merge(res_user_hyperedges, unique_user_id, how='left', on='user_id_matrix')
    res_user_hyperedges = pd.merge(res_user_hyperedges, user_play_count, how='left', on='user_id_matrix')
    return res_user_hyperedges[['user_id', 'tracks', 'play_count']].copy()


def album_tracks_hyperedges_construction(para_track_data, para_album_data):
    res_album_hyperedges = para_track_data.groupby('album_id_matrix')['track_id'].apply(list).reset_index(
        name='tracks')
    res_album_hyperedges['album_id_matrix'] = res_album_hyperedges['album_id_matrix'].astype('int32')
    album_and_play_count = para_track_data.groupby('album_id_matrix')['play_count'].apply(list).reset_index(
        name='play_count')
    album_and_play_count['album_id_matrix'] = album_and_play_count['album_id_matrix'].astype('int32')
    res_album_hyperedges = pd.merge(res_album_hyperedges,
                                    para_album_data[['album_id', 'album_id_matrix']], how='left', on='album_id_matrix')
    res_album_hyperedges = pd.merge(res_album_hyperedges, album_and_play_count, how='left', on='album_id_matrix')
    return res_album_hyperedges[['album_id', 'tracks', 'play_count']].copy()


def artist_tracks_hyperedges_construction(para_track_data, para_artist_data):
    res_artist_hyperedges = para_track_data.groupby('artist_id_matrix')['track_id'].apply(list).reset_index(
        name='tracks')
    res_artist_hyperedges = pd.merge(res_artist_hyperedges, para_artist_data[['artist_id_matrix', 'artist_id']],
                                     how='left', on='artist_id_matrix')
    track_and_play_count = para_track_data.groupby('artist_id_matrix')['play_count'].apply(list).reset_index(
        name='play_count')
    res_artist_hyperedges = pd.merge(res_artist_hyperedges, track_and_play_count, how='left', on='artist_id_matrix')
    return res_artist_hyperedges[['artist_id', 'tracks', 'play_count']].copy()


def tag_hyperedges_construction(para_track_data):
    tag_dict = {}

    def inner_add_tag_edges(tag_track_row):
        tags = eval(tag_track_row[1])
        for tag_info in tags:
            temp_tag = tag_info[0]
            count = tag_info[1]
            if temp_tag not in tag_dict:
                tag_dict[temp_tag] = {}
                tag_dict[temp_tag]['tracks'] = []
                tag_dict[temp_tag]['tag_id'] = 'ta_' + str(len(tag_dict) - 1)
                tag_dict[temp_tag]['count'] = []

            tag_dict[temp_tag]['count'].append(count)
            tag_dict[temp_tag]['tracks'].append(tag_track_row[0])

    para_track_data[['track_id', 'tags']].apply(inner_add_tag_edges, axis=1)
    res_tag_hyperedges = pd.DataFrame.from_dict(tag_dict, orient='index')
    res_tag_hyperedges.reset_index(drop=True, inplace=True)
    return res_tag_hyperedges[['tag_id', 'tracks', 'count']].copy()


# hyperedge users
def add_user_listening_tracks_edges(para_user_row, para_offset):
    para_user_row['v_members'] = np.zeros(shape=(len(para_user_row['tracks']) + 1), dtype=object)
    para_user_row['v_weights'] = np.zeros(shape=(len(para_user_row['play_count']) + 1), dtype=float)

    para_user_row['v_members'][0] = para_user_row['user_id']
    para_user_row['v_members'][1:] = para_user_row['tracks']
    para_user_row['v_weights'][0] = 0.5
    para_user_row['v_weights'][1:] = np.array(para_user_row['play_count']) / np.sum(para_user_row['play_count']) * 0.5

    para_user_row['e_id'] = 'e_' + str(para_offset + int(para_user_row['user_id'][2:]))
    para_user_row['id'] = para_user_row['e_id']
    para_user_row['e_type'] = 'user'
    para_user_row['e_weight'] = 1.0
    return para_user_row[['id', 'e_id', 'e_type', 'e_weight', 'v_members', 'v_weights']]


def add_tag_edges(para_tag_row, para_offset):
    para_tag_row['v_members'] = np.zeros(shape=(len(para_tag_row['tracks']) + 1,), dtype=object)
    para_tag_row['v_weights'] = np.zeros(shape=(len(para_tag_row['count']) + 1,), dtype=float)

    para_tag_row['v_members'][0] = para_tag_row['tag_id']
    para_tag_row['v_members'][1:] = para_tag_row['tracks']

    para_tag_row['v_weights'][0] = 0.5
    para_tag_row['v_weights'][1:] = np.array(para_tag_row['count']) / np.sum(para_tag_row['count']) * 0.5

    para_tag_row['e_id'] = 'e_' + str(para_offset + int(para_tag_row['tag_id'][3:]))
    para_tag_row['id'] = para_tag_row['e_id']
    para_tag_row['e_type'] = 'tag'
    para_tag_row['e_weight'] = 1.0
    return para_tag_row[['id', 'e_id', 'e_type', 'e_weight', 'v_members', 'v_weights']]


def add_album_containing_tracks_edges(para_album_row, para_offset):
    para_album_row['v_members'] = np.zeros(shape=(len(para_album_row['tracks']) + 1,), dtype=object)
    para_album_row['v_weights'] = np.zeros(shape=(len(para_album_row['play_count']) + 1,), dtype=float)

    para_album_row['v_members'][0] = para_album_row['album_id']
    para_album_row['v_members'][1:] = para_album_row['tracks']
    para_album_row['v_weights'][0] = 0.5
    para_album_row['v_weights'][1:] = np.array(para_album_row['play_count']) / np.sum(
        para_album_row['play_count']) * 0.5

    para_album_row['e_id'] = 'e_' + str(para_offset + int(para_album_row['album_id'][3:]))
    para_album_row['id'] = para_album_row['e_id']
    para_album_row['e_type'] = 'album'
    para_album_row['e_weight'] = 1.0
    return para_album_row[['id', 'e_id', 'e_type', 'e_weight', 'v_members', 'v_weights']]


def add_artist_containing_tracks_edges(para_artist_row, para_offset):
    para_artist_row['v_members'] = np.zeros(shape=(len(para_artist_row['tracks']) + 1,), dtype=object)
    para_artist_row['v_weights'] = np.zeros(shape=(len(para_artist_row['play_count']) + 1,), dtype=float)

    para_artist_row['v_members'][0] = para_artist_row['artist_id']
    para_artist_row['v_members'][1:] = para_artist_row['tracks']
    para_artist_row['v_weights'][0] = 0.5
    para_artist_row['v_weights'][1:] = np.array(para_artist_row['play_count']) / np.sum(
        para_artist_row['play_count']) * 0.5

    para_artist_row['e_id'] = 'e_' + str(para_offset + int(para_artist_row['artist_id'][3:]))
    para_artist_row['id'] = para_artist_row['e_id']
    para_artist_row['e_type'] = 'artist'
    para_artist_row['e_weight'] = 1.0
    return para_artist_row[['id', 'e_id', 'e_type', 'e_weight', 'v_members', 'v_weights']]


def calculate_category_distribution(para_hyperedges):
    cat_amounts = {}
    for edge in para_hyperedges:
        if para_hyperedges[edge]['e_type'] not in cat_amounts:
            cat_amounts[para_hyperedges[edge]['e_type']] = 0
        cat_amounts[para_hyperedges[edge]['e_type']] += 1
    return cat_amounts


def draw_category_distribution(para_hyperedges):
    cat_amounts = calculate_category_distribution(para_hyperedges)
    print(cat_amounts)

    # plot hyperedges distribution
    pd_df = pd.DataFrame(list(cat_amounts.items()))
    pd_df.columns = ['Dim', 'Count']
    # sort df by Count column
    pd_df = pd_df.sort_values(['Count']).reset_index(drop=True)
    plt.bar(pd_df['Dim'], pd_df['Count'])
    for _, row in pd_df.iterrows():
        cat = row['Dim']
        count = row['Count']
        plt.text(cat, count + 0.1, str(count), ha='center', va='bottom')

    plt.show()


def hyperedges_construction_phase(para_inter_data, para_track_data, para_album_data, para_artist_data,
                                  para_control_args):
    if not para_control_args['model']['hypergraph']:
        return None, None
    print(' '.join(['Fold:', str(para_control_args['dataset']['fold']), 'hypergraph constructing.']))
    hyperedges = dict()
    # Creating hypergraph -- Build the listening events' hyperedge (user-track $e^{(1)}$).
    user_tracks_data = user_tracks_hyperedges_construction(para_inter_data)
    u_edges = user_tracks_data.apply(add_user_listening_tracks_edges, args=(0,), axis=1)
    hyperedges.update(u_edges.set_index('id').to_dict(orient='index'))

    # Creating hypergraph -- Build the tag's hyperedge (tag-track hyperedge $e^{(2)}$).
    if para_control_args['ablation']['tag-track']:
        tag_tracks_data = tag_hyperedges_construction(para_track_data)
        ta_edges = tag_tracks_data.apply(add_tag_edges, args=(len(hyperedges),), axis=1)
        hyperedges.update(ta_edges.set_index('id').to_dict(orient='index'))

    # Creating hypergraph -- Build the album-tracks hyperedge (album-track hyperedge $e^{(3)}$).
    if para_control_args['ablation']['album-track']:
        album_tracks_data = album_tracks_hyperedges_construction(para_track_data, para_album_data)
        al_edges = album_tracks_data.apply(add_album_containing_tracks_edges, args=(len(hyperedges),), axis=1)
        hyperedges.update(al_edges.set_index('id').to_dict(orient='index'))

    # Creating hypergraph -- Build the artist's hyperedge (artist-track hyperedge $e^{(4)}$).
    if para_control_args['ablation']['artist-track']:
        artist_tracks_data = artist_tracks_hyperedges_construction(para_track_data, para_artist_data)
        ar_edges = artist_tracks_data.apply(add_artist_containing_tracks_edges, args=(len(hyperedges),), axis=1)
        hyperedges.update(ar_edges.set_index('id').to_dict(orient='index'))

    # count hyperedge number per category
    # calculate_category_distribution(hyperedges)

    # draw_category_distribution(hyperedges)

    # num_hyperedges = len(hyperedges)
    # num_vertices = num_hyperedges + len(para_track_data)
    # incidence_matrix = np.zeros(shape=(num_vertices, num_hyperedges))
    # vertices_array = np.zeros(shape=(num_vertices,), dtype=object)
    # hyperedges_array = np.zeros(shape=(num_hyperedges,), dtype=object)
    # hyperedges_weight_array = np.zeros(shape=(num_hyperedges,), dtype=object)
    #
    # p = 0
    # for head in hyperedges:
    #     he = Hyperedge(hyperedges[head]['category'], head, hyperedges[head]['members'][:-1],
    #                    hyperedges['head']['weights'][-1])
    #     hyperedges_array[p] = he
    #     p += 1
    #
    # Hypergraph(vertices_array, )

    # for each node, get hyperedges
    vertex_links = {}
    for edge in hyperedges:
        vertices = hyperedges[edge]['v_members']
        for vertex in vertices:
            if vertex in vertex_links:
                vertex_links[vertex].append(edge)
            else:
                vertex_links[vertex] = [edge]
    # hyperedges = normalize_weight(hyperedges)
    return hyperedges, vertex_links
