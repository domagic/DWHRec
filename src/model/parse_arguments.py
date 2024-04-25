import argparse


def get_arguments():
    """
    Get arguments from the console.
    :return: The console arguments.
    """
    parser = argparse.ArgumentParser()
    # add console arguments
    parser.add_argument('--dataset', type=str, default='100k', help='name of dataset')
    parser.add_argument('--ph', action='store_false', help='hypergraph construction phase')
    parser.add_argument('--pw', action='store_false', help='random walk phase')
    parser.add_argument('--pe', action='store_false', help='vertex embedding phase')
    parser.add_argument('--pr', action='store_false', help='recommendation phase')
    parser.add_argument('--e2', action='store_false', help='tag-track hyperedge')
    parser.add_argument('--e3', action='store_false', help='album-track hyperedge')
    parser.add_argument('--e4', action='store_false', help='artist-track hyperedge')
    parser.add_argument('--r', type=int, default=1, help='number of iterations in random walks')
    parser.add_argument('--k', type=int, default=100, help='number of steps in each iteration')
    parser.add_argument('--s', type=int, default=100, help='size of embedding vector')
    parser.add_argument('--w', type=int, default=5, help='window size of skip-gram')
    parser.add_argument('--n', type=int, default=100, help='length of top-n recommendation list')
    # parse console arguments
    args = parser.parse_args()
    return args
