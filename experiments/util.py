import jax
import jax.numpy as jnp
import jraph
import numpy as np
import networkx as nx


def compute_node_degrees(n_nodes, senders):
    """ Compute degrees of each node in [0,...,n_nodes-1]
    :param n_nodes: number of nodes of the graph
    :param senders: array of length number_of_edges with outgoing nodes for each edge
    """
    degrees = jax.vmap(lambda n: jnp.count_nonzero(senders == n))(jnp.arange(n_nodes))
    return np.array(degrees)


def convert_jraph_to_networkx_graph(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
    nodes, edges, receivers, senders, _, _, _ = jraph_graph
    nodes = np.array(nodes)
    edges = np.array(edges)
    receivers = np.array(receivers)
    senders = np.array(senders)
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(jraph_graph.n_node[0]), node_feature=nodes)
    nx_graph.add_weighted_edges_from(zip(senders, receivers, edges[:, 0]))
    return nx_graph


def spd_one_hot(n_nodes: int) -> np.ndarray:
    # smallest integer d such that size of upper triangle without diagonal greater than n_nodes
    d = np.ceil(-1 / 2 + np.sqrt(1 + 8 * n_nodes) / 2).astype(int) + 1

    ind_utr = np.triu_indices(d)

    P = jnp.zeros((n_nodes, d, d))
    for i in range(n_nodes):
        P = P.at[i, ind_utr[0][i], ind_utr[1][i]].set(1)
        P = P.at[i, ind_utr[1][i], ind_utr[0][i]].set(1)

    return P


def convert_networkx_to_jraph_graph(nx_graph: nx.Graph,
                                    space: str,
                                    features: str,
                                    edge_weights: jnp.ndarray | None,
                                    global_feature: jnp.ndarray | None = None,
                                    dim: int | None = None) -> jraph.GraphsTuple:

    assert space == "hyperbolic" or space == "spd"
    assert features == "one_hot" or features == "degree"

    e = jnp.array(nx_graph.edges)

    senders = e.ravel()
    receivers = e[:, ::-1].ravel()

    n_nodes = nx_graph.number_of_nodes()

    if space == "hyperbolic":
        if features == "degree":
            degrees = compute_node_degrees(n_nodes, senders)

            if dim is None:
                dim = jnp.max(degrees)
            elif dim < jnp.max(degrees):
                print("Warning: The outer dimension is smaller than the maximal degree. All nodes with a larger degree than the outer dimension are mapped to the pole.")

            f = jax.nn.one_hot(degrees, dim)
        else:  # features == "one_hot"
            f = jnp.eye(n_nodes + 1)[:n_nodes]
    else:  # if space == "spd"
        if features == "degree":
            raise NotImplementedError('Degree representation for the SPD manifold has not been implemented yet.')

        else:  # features == "one_hot"
            f = spd_one_hot(n_nodes)

    if edge_weights is None:
        edge_weights = jnp.ones_like(senders)[:, None] / nx_graph.number_of_nodes()
    else:
        assert len(edge_weights) == len(nx_graph.edges)

    jgraph = jraph.GraphsTuple(
        nodes=f,
        edges=edge_weights,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([nx_graph.number_of_nodes()]),
        n_edge=jnp.array([2 * nx_graph.number_of_edges()]),
        globals=global_feature
    )

    return jgraph



