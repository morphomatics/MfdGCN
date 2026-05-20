from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import jraph

from typing import List, Generator, Tuple

import numpy as np
import flax.linen as nn
import optax

from morphomatics.manifold import HyperbolicSpace, Sphere, SPD, Euclidean
from morphomatics.nn.flow_layers import FlowLayer, MfdGcnBlock
from morphomatics.nn.wFM_layers import MfdInvariant
from morphomatics.nn.euclidean_layers import MLP
from morphomatics.graph.operators import max_pooling, mean_pooling

from data.synthetic_graphs.generate_graphs_morphomatics import generate_graphs, pickle_load
from util import spd_one_hot
from train import update, evaluate_f1, TrainingState

NUM_CLASSES = 3
NUM_NODES = 50


def batch_iterate(data: List[jraph.GraphsTuple],
                  batch_size: int,
                  num_edges) -> Generator[jraph.GraphsTuple, None, None]:

    # plus one for the extra padding node.
    return jraph.dynamically_batch(data, n_node=batch_size * NUM_NODES + 1, n_edge=num_edges, n_graph=batch_size + 1)


def generate_data(graph_num: int,
                  feature_space: str,
                  feature_initialization: str,
                  batch_size: int,
                  n_val: int,
                  n_test: int,
                  key: jnp.ndarray) -> Tuple[List[jraph.GraphsTuple], List[jraph.GraphsTuple], List[jraph.GraphsTuple]]:

    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        if graph_num == 2000:
            # compare all methods on the same data set (and split)
            data = pickle_load('../data/synthetic_graphs/synthetic_data_random.pkl')
        else:
            data = generate_graphs(NUM_NODES, NUM_NODES, graph_num, key, feature_space, feature_initialization)

        if feature_space == "spd":
            f = spd_one_hot(NUM_NODES)

            for i, G in enumerate(data):
                data[i] = G._replace(nodes=f)

        # maximal number of edges in batching
        num_edges = NUM_NODES ** 2 * batch_size
        batched_data = list(batch_iterate(data, batch_size, num_edges))

        n_train = graph_num - n_val - n_test
        data_train = batched_data[:n_train]
        data_val = batched_data[n_train: n_train + n_val]
        data_test = batched_data[n_train + n_val:]

    return data_train, data_val, data_test


def train(network: nn.Module,
          optimizer: optax.GradientTransformation,
          data: Tuple[List[jraph.GraphsTuple], List[jraph.GraphsTuple], List[jraph.GraphsTuple]],
          n_epochs: int,
          key: jnp.ndarray)-> Tuple[float, jnp.ndarray, jnp.ndarray]:

    data_train, data_val, data_test = data

    key, subkey = jax.random.split(key)
    initial_params = network.init(subkey, data_train[0])

    initial_opt_state = optimizer.init(initial_params)
    state = TrainingState(initial_params, initial_params, initial_opt_state)

    train_accuracies = []
    test_accuracies = []
    opt_val = 0.
    opt_test = 0.

    for epoch in range(n_epochs):

        # training & evaluation loop.
        for step, batch in enumerate(data_train):
            key, subkey = jax.random.split(key)

            # do SGD on a batch of training examples.
            mask = jraph.get_graph_padding_mask(batch)
            state = update(state=state,
                           graph=batch,
                           label=batch.globals,
                           optimizer=optimizer,
                           network=network,
                           mask=mask)

        train_accuracy = 0.
        for i, batch in enumerate(data_train):
            mask = jraph.get_graph_padding_mask(batch)
            train_accuracy += evaluate_f1(params=state.avg_params,
                                          graph=batch,
                                          labels=batch.globals,
                                          num_classes=NUM_CLASSES,
                                          network=network,
                                          mask=mask)

        train_accuracy /= len(data_train)

        train_accuracies.append(train_accuracy)

        validation_accuracy = 0.
        for i, batch in enumerate(data_val):
            mask = jraph.get_graph_padding_mask(batch)
            validation_accuracy += evaluate_f1(params=state.avg_params,
                                               graph=batch,
                                               labels=batch.globals,
                                               num_classes=NUM_CLASSES,
                                               network=network,
                                               mask=mask)

        validation_accuracy /= len(data_val)

        test_accuracy = 0.
        for i, batch in enumerate(data_test):
            mask = jraph.get_graph_padding_mask(batch)
            test_accuracy += evaluate_f1(params=state.avg_params,
                                         graph=batch,
                                         labels=batch.globals,
                                         num_classes=NUM_CLASSES,
                                         network=network,
                                         mask=mask)

        test_accuracy /= len(data_test)

        test_accuracies.append(test_accuracy)

        # update criterion for the to-be-reported test result
        if validation_accuracy >= opt_val:
            opt_val = validation_accuracy
            opt_test = test_accuracy

    return opt_test, jnp.array(train_accuracies), jnp.array(test_accuracies)


class FlowNetwork(nn.Module):
    M: Manifold
    depth: int
    width: int
    feature_initialization: str
    max_step_length: float


    @nn.compact
    def __call__(self, G: jraph.GraphsTuple) -> jnp.ndarray:

        z = G.nodes

        if self.feature_initialization == "degree" and isinstance(self.M, HyperbolicSpace):
            to_tan = nn.Dense(NUM_NODES, use_bias=False)
            z = jax.vmap(to_tan)(z)

            # interpret z as an element of the tangent space at p
            z = jnp.concatenate((z, jnp.zeros((len(z), 1))), axis=1)

        if isinstance(self.M, SPD):
            # smallest integer d such that num_nodes <= dim(SPD(d))
            p = jnp.eye(self.M._d)
        else:
            p = jax.nn.one_hot(NUM_NODES, NUM_NODES + 1)

        z = jax.vmap(lambda v: self.M.connec.exp(p, v))(z)[:, None, :]

        G = G._replace(nodes=jnp.concatenate([z, ] * 5, axis=1))

        G = MfdGcnBlock(self.M, [self.width, ] * self.depth, max_step_length=self.max_step_length)(G)
        G = FlowLayer(self.M, max_step_length=self.max_step_length)(G)

        z = MfdInvariant(self.M, self.width, nC=2)(G.nodes[None])[0]
        z = jax.nn.leaky_relu(z)
        z = jnp.concatenate((max_pooling(G, z), mean_pooling(G, z)), axis=1)

        # MLP mapping to NUM_CLASSES channels per graph
        f = MLP((self.width, self.width // 2, NUM_CLASSES))

        return f(z)


def main(n_graphs: int,
         batch_size: int,
         n_val: int,
         n_test: int,
         n_epochs: int,
         feature_space: str,
         feature_initialization: str,
         n_layers: int,
         n_channels: int,
         seed: jnp.ndarray) -> float:

    if feature_space == "hyperbolic":
        M = HyperbolicSpace((NUM_NODES + 1,))
        max_step_length = 1.
    else:  # feature_space == "spd":
        # smallest integer d such that num_nodes <= dim(SPD(d))
        d = np.ceil(-1 / 2 + np.sqrt(1 + 8 * NUM_NODES) / 2).astype(int) + 1
        M = SPD(d, structure="AffineInvariant")
        max_step_length = 1.

    data = generate_data(n_graphs, feature_space, feature_initialization, batch_size, n_val, n_test, seed)

    # make the network and optimizer
    network = FlowNetwork(M, n_layers, n_channels, feature_initialization, max_step_length)
    optimizer = optax.adam(1e-3)

    test_f1, _, _ = train(network, optimizer, data, n_epochs, jax.random.key(42))

    return test_f1


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    num_graphs = int(sys.argv[1])
    space = sys.argv[2]  # "hyperbolic" or "spd"
    initialization_type = sys.argv[3]  # "one_hot" or "degree"
    n_repeats = int(sys.argv[4])

    num_layers = 2
    num_channels = 16

    assert num_graphs  > 0
    assert space in ["hyperbolic", "spd"]
    assert initialization_type in ["one_hot", "degree"]

    size_batches= 3
    num_val = int(num_graphs / 6)
    num_test = int(num_graphs / 6)
    num_epochs = 60

    results = []

    print(f"\nStart training on {3 * num_graphs} graphs.")
    for s in range(n_repeats):
        results.append(
            main(n_graphs=num_graphs,
                 batch_size=size_batches,
                 n_val=num_val,
                 n_test=num_test,
                 n_epochs=num_epochs,
                 feature_space=space,
                 feature_initialization=initialization_type,
                 n_layers=num_layers,
                 n_channels=num_channels,
                 seed=jax.random.key(s)))

        print(f"\nF1 score seed {s}: {results[-1]:.3f}")
        print(f"Running average: {np.mean(results):.3f}")

    print(f"\nAverage F1 score: {np.mean(results):.3f}, Standard Deviation: {np.std(results):.3f}")