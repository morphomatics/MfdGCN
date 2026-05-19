from typing import List

import jax
import jax.numpy as jnp

import jraph

import networkx as nx
import numpy as np
import pickle


from util import convert_networkx_to_jraph_graph


def pickle_dump(file_name, content):
    with open(file_name, 'wb') as out_file:
        pickle.dump(content, out_file, pickle.HIGHEST_PROTOCOL)


def pickle_load(file_name):
    with open(file_name, 'rb') as in_file:
        return pickle.load(in_file)


def generate_graphs(max_node_num: int,
                    min_node_num: int,
                    graph_num: int,
                    key: jnp.ndarray, space: str,
                    features: str,
                    save_pickle: bool = False) -> List[jraph.GraphsTuple]:
    """
    Generate a random graph using 3 different algorithms: the Erdös-Renyi (E-R), Watts-Strogatz (W-S), and
    Barabasi-Albert (B-A) algorithms.
    :param max_node_num: maximal number of nodes a graph may have
    :param min_node_num: minimal number of nodes a graph may have
    :param graph_num: number of graphs of each type
    :param key: PRNG key generating the pseudo-random sequence
    :param space: string 'hyperbolic', 'sphere', or 'SPD' deciding whether features are encoded as vectors or matrices
    :param features: string 'degree' or 'one_hot' indicating which features are to be used
    :param save_pickle: if true the graphs are saved using picke
    :return: list of graphs ordered as (E-R, W-S, B-R,...,E-R, W-S, B-R)
    """

    data = []
    for i in range(graph_num):
        key, subkey1, subkey2 = jax.random.split(key, 3)

        ### Erdös-Renyi ###

        num_node = jax.random.randint(subkey1, (1,), min_node_num, max_node_num)[0]
        graph = nx.erdos_renyi_graph(num_node, jax.random.uniform(subkey2, (1,), minval=0.01, maxval=1)[0])
        erdos_renyi_graph = convert_networkx_to_jraph_graph(graph, space=space, features=features, edge_weights=None,
                                                            global_feature=jnp.array([0], int), dim=max_node_num)

        ### Watts-Strogatz (small world) ###

        while True:
            key, subkey1, subkey2 = jax.random.split(key, 3)
            try:
                num_node = jax.random.randint(subkey1, (1,), min_node_num, max_node_num)[0]
                graph = nx.watts_strogatz_graph(num_node, np.random.randint(low=1, high=200),
                                                jax.random.uniform(subkey2, (1,), minval=0.01, maxval=1)[0])
                small_world_graph = convert_networkx_to_jraph_graph(graph, space=space, features=features,
                                                                    edge_weights=None,
                                                                    global_feature=jnp.array([1], int),
                                                                    dim=max_node_num)

                break
            except:
                pass

        ### Barabasi-Albert ###

        while True:
            key, subkey1, subkey2 = jax.random.split(key, 3)
            try:
                num_node = jax.random.randint(subkey1, (1,), min_node_num, max_node_num)[0]
                graph = nx.barabasi_albert_graph(num_node,
                                                 int(jax.random.randint(subkey2, (1,), minval=1, maxval=200)[0]))
                barabasi_albert_graph = convert_networkx_to_jraph_graph(graph, space=space, features=features,
                                                                        edge_weights=None,
                                                                        global_feature=jnp.array([2], int),
                                                                        dim=max_node_num)

                break
            except:
                pass

        data.extend((erdos_renyi_graph, small_world_graph, barabasi_albert_graph))

    if save_pickle:
        pickle_dump('synthetic_data_random.pkl', data)

    return data
