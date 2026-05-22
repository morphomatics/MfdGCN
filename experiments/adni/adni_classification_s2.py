from glob import glob
import argparse
from typing import NamedTuple, List, Generator
import sys

sys.path.insert(0, "../..")

import numpy as np
import scipy
from sklearn import model_selection
import jax
import jax.numpy as jnp
import pyvista as pv

import jraph
import optax
import flax.linen as nn

from morphomatics.nn.flow_layers import FlowLayer, TangentMLP
from morphomatics.nn.wFM_layers import MfdInvariant
from morphomatics.nn.euclidean_layers import MLP
from morphomatics.graph.operators import max_pooling, mean_pooling
from morphomatics.geom import Surface
from morphomatics.manifold import Sphere

from train import update, evaluate, TrainingState
from adni_classification_shape_space import batch_iterate

###########################################
# Data
###########################################
NUM_CLASSES = 2
MAX_NUM_NODES = -1
MAX_NUM_EDGES = -1
MAX_VOLUME = 0


class Hippocampus(NamedTuple):
    mesh: pv.PolyData
    W: scipy.sparse.coo_matrix
    y: int

def load_meshes():
    # read
    hippocampi: List[Hippocampus] = []
    files = glob('../../data/adni/**/*obj', recursive=True)
    for f in files:
        obj: pv.PolyData = pv.read(f)
        if obj.volume > MAX_VOLUME:
            globals()['MAX_VOLUME'] = obj.volume
        obj.compute_normals(inplace=True, cell_normals=False)

        # (weak) Laplacian
        surf = Surface(obj.points, obj.faces.reshape(-1, 4)[:, 1:])
        L = (surf.div @ surf.grad).tocoo()

        # to edge weights
        L.setdiag(0)
        L.eliminate_zeros()
        # weight normalization
        D = L * np.ones(L.shape[1])
        L = L.tocoo()
        L.data /= -np.sqrt(np.abs(D[L.col] * D[L.row]))
        L.data = np.clip(L.data, 1e-6, None)

        y = int(f.split('/')[-2] == 'AD')

        global MAX_NUM_NODES, MAX_NUM_EDGES

        hippocampi.append(Hippocampus(obj, L, y))
        MAX_NUM_NODES = max(MAX_NUM_NODES, obj.n_points)
        MAX_NUM_EDGES = max(MAX_NUM_EDGES, L.nnz // 2)
    assert len(np.unique([h.y for h in hippocampi])) == NUM_CLASSES

    return hippocampi


def iterate(data: List[Hippocampus]) -> Generator[jraph.GraphsTuple, None, None]:
    for hippo in data:
        x = jnp.asarray(hippo.mesh['Normals'], dtype=jnp.float64)[..., None, :]
        W = hippo.W
        yield jraph.GraphsTuple(
            n_node=jnp.asarray([len(x)]),
            n_edge=jnp.asarray([W.nnz]),
            nodes=x,
            edges=jnp.asarray(W.data, dtype=jnp.float64),
            globals=jnp.array([[hippo.mesh.volume / MAX_VOLUME, hippo.y]]),
            senders=jnp.asarray(W.row),
            receivers=jnp.asarray(W.col))


###########################################
# GCN
###########################################
class GCN(nn.Module):
    depth: int
    width: int

    @nn.compact
    def __call__(self, G: jraph.GraphsTuple) -> jnp.ndarray:
        Gs = [G._replace(nodes=G.nodes[:, 0]) for _ in range(self.width)]

        for l in range(self.depth):
            for i, G in enumerate(Gs):
                G = jraph.GraphConvolution(update_node_fn=lambda n: jax.nn.relu(nn.Dense(3)(n)),
                                           add_self_edges=True)(G)
                Gs[i] = G

            z = jnp.hstack([G.nodes for G in Gs])
            out = 3 * self.width if l < 2 else self.width
            z = jax.nn.relu(nn.Dense(out)(z)).reshape(len(z), self.width, -1)
            for i, G in enumerate(Gs):
                Gs[i] = G._replace(nodes=z[:, i])

        z = jnp.hstack([G.nodes for G in Gs])

        # global pooling
        z = jnp.concatenate((max_pooling(G, z), mean_pooling(G, z)), axis=1)
        # add volume information
        z = jnp.append(z, jnp.atleast_2d(G.globals[:, 0]).T, axis=1)

        ### MLP mapping to NUM_CLASSES channels per graph ###
        return MLP((3, 3, NUM_CLASSES))(z)


###########################################
# Manifold GCN
###########################################
class MfdGCN(nn.Module):
    depth: int
    width: int

    @nn.compact
    def __call__(self, G: jraph.GraphsTuple) -> jnp.ndarray:
        n_steps = 1

        # signal domain
        M = Sphere()

        G = G._replace(nodes=jnp.concatenate([G.nodes, ] * self.width, axis=1))

        for i in range(self.depth - 1):
            G = FlowLayer(M, n_steps)(G)
            G = G._replace(nodes=TangentMLP(M, (self.width,))(G.nodes[None])[0])

        G = FlowLayer(M, n_steps)(G)

        # invariant layer
        z = MfdInvariant(M, self.width)(G.nodes[None])[0]
        z = jax.nn.leaky_relu(z)

        # global pooling
        z = jnp.concatenate((max_pooling(G, z), mean_pooling(G, z)), axis=1)
        # add volume information
        z = jnp.append(z, jnp.atleast_2d(G.globals[:, 0]).T, axis=1)

        # MLP mapping to NUM_CLASSES channels per graph
        return MLP((self.width, self.width // 2, NUM_CLASSES))(z)


def training(hippocampi: List[Hippocampus],
             network: nn.Module,
             optimizer: optax.GradientTransformation,
             batch_size: int,
             n_epochs: int,
             seed: int,
             verbosity: int = 0) -> float:
    """Train and trest procedure. Splits the ADNI data into training, validation, and test sets. The validation set is
    used to select the best model while avoiding overfitting the data. The performance of the selected model on the test
    data is returned.

    :param hippocampi: ADNI data
    :param network: network to train
    :param optimizer: optimizer to use
    :param batch_size: Batch size
    :param n_epochs: Number of epochs
    :param seed: Random seed
    :param verbosity: 0, 1 verbosity level
    :return: accuracy on test data
    """

    # split data in training and testing sets
    data_train_full, data_test = model_selection.train_test_split(
        hippocampi, test_size=0.2, random_state=seed, stratify=[h.y for h in hippocampi])

    # hold back validation set from the training data
    data_train, data_validation = model_selection.train_test_split(
        data_train_full, test_size=0.25, random_state=seed, stratify=[h.y for h in data_train_full])

    # evaluate accuracy function
    eval_ = lambda p, g: evaluate(p, g, g.globals[:, 1], num_classes=NUM_CLASSES, network=network,
                                  mask=jnp.ones((1,)))

    key = jax.random.key(0)
    params = network.init(key, next(batch_iterate(hippocampi, batch_size)))
    if verbosity >= 1:
        flat_para, _ = jax.flatten_util.ravel_pytree(params)
        print(f"\nNumber of network parameters: {len(flat_para)}")

    # initialize optimizer state
    opt_state = optimizer.init(params)
    state = TrainingState(params, params, opt_state)

    opt_acc = 0.
    opt_param = state.params
    # training loop
    for i in range(n_epochs):
        for step, batch in enumerate(batch_iterate(data_train, batch_size)):
            mask = jraph.get_graph_padding_mask(batch)
            state = update(state, batch, batch.globals[:, 1], optimizer, network, mask)

        # evaluate accuracy (no batching/padding -> no masking)
        validation_acc = np.mean([eval_(state.avg_params, g) for g in iterate(data_validation)])
        _test_acc = np.mean([eval_(state.avg_params, g) for g in iterate(data_test)])

        # update optimal parameters (only after epoch 25 to ignore randomly high validation accuracy early in the training)
        if validation_acc > opt_acc and i > 25:
            opt_param = state.avg_params
            # update optimal validation accuracy
            opt_acc = validation_acc

    test_acc = np.mean([eval_(opt_param, g) for g in iterate(data_test)])

    return test_acc


def main(case: str,
         seed: int):

    # hyperparameters
    learning_rate = 1e-3
    batch_size = 1
    n_epochs = 300

    hippocampi = load_meshes()

    # use a comparable number of parameters
    if case == "mfdgcn":
        # initialize network
        network = MfdGCN(4, 16)

        transition_steps = len(hippocampi) // batch_size
        decay_rate = 0.95
        transition_begin = 0

        schedule = optax.exponential_decay(learning_rate, transition_steps, decay_rate, transition_begin)
        optimizer = optax.adam(learning_rate=schedule)
    elif case == "gcn":
        network = GCN(3, 16)

        optimizer = optax.adam(learning_rate=learning_rate)

    acc = training(hippocampi=hippocampi,
                   network=network,
                   optimizer=optimizer,
                   batch_size=batch_size,
                   n_epochs=n_epochs,
                   seed=seed)

    return acc


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--network",
                        type=str, help="whether to use Manifold GCN or a standard GCN, indicate by 'mfdgcn' or 'gcn' ",
                        default="mfdgcn")
    args = parser.parse_args()
    case = args.network

    net_str = "MfdGCN" if case == "flow" else "GCN"
    print(f"\nRunning {net_str} on ADNI data using normals with 100 random seeds...")
    results = []
    for s in range(100):
        results.append(main(case, s))
        print(f"\nSeed {s}: {results[-1]:.3f}")
        print(f"Running average: {np.mean(np.array(results)):.3f}")

    results = np.array(results)
    print(f"\nAverage accuracy {np.mean(results):.3f}, Standard deviation {np.std(results):.3f}")

