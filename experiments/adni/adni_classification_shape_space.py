from typing import NamedTuple, List, Generator
import sys

sys.path.insert(0, "../..")

import numpy as np

from sklearn import model_selection

import jax
import jax.numpy as jnp

import jraph
import optax
import flax.linen as nn

from morphomatics.graph import max_pooling, mean_pooling
from morphomatics.nn.flow_layers import MfdGcnBlock, FlowLayer
from morphomatics.nn.wFM_layers import MfdInvariant
from morphomatics.nn.euclidean_layers import MLP
from morphomatics.geom import Surface
from morphomatics.manifold import SO3, SPD

from experiments.train import update, evaluate, TrainingState
from glob import glob
import os.path as osp
import pyvista as pv

###########################################
# Data
###########################################
NUM_CLASSES = 2
MAX_NUM_NODES = -1
MAX_NUM_EDGES = -1
MAX_VOLUME = 0


class Hippocampus(NamedTuple):
    features: jnp.ndarray  # differential coordinates
    e: jnp.ndarray  # edges
    w: jnp.ndarray  # weight vector
    volume: float  # mesh volume
    y: int  # class indicator


def read_data():
    # load differential coordinates
    coords = np.load('../../data/adni/dcm_coords.npz')

    hippocampi: List[Hippocampus] = []
    for f in glob('../../data/adni/**/*obj', recursive=True):
        sid = osp.basename(f)[:-4]
        obj = pv.read(f)
        if obj.volume > MAX_VOLUME:
            globals()['MAX_VOLUME'] = obj.volume

        surf = Surface(obj.points, obj.faces.reshape(-1, 4)[:, 1:])
        _, _, _, n_1, _, _ = surf.neighbors

        n = len(surf.f)
        # edges of the dual mesh
        e = np.hstack(([np.arange(n), n_1[:, 0]],
                        [np.arange(n), n_1[:, 1]],
                        [np.arange(n), n_1[:, 2]]))

        y = int(f.split('/')[-2] == 'AD')

        hippocampi.append(Hippocampus(features=np.swapaxes(coords[sid], 0, 1),
                                      e=e.T,
                                      w=np.ones((len(e.T), 1)) / len(e.T),
                                      volume=obj.volume,
                                      y=y))
        globals()['MAX_NUM_NODES'] = max(MAX_NUM_NODES, obj.n_cells)
        globals()['MAX_NUM_EDGES'] = max(MAX_NUM_EDGES, len(e.T) // 2)
    assert len(np.unique([h.y for h in hippocampi])) == NUM_CLASSES

    return hippocampi


def iterate(data: List[Hippocampus]) -> Generator[jraph.GraphsTuple, None, None]:
    for hippo in data:
        C = hippo.features
        e = hippo.e
        w = hippo.w
        yield jraph.GraphsTuple(
            n_node=np.array([len(C)]),
            n_edge=np.array([len(e)]),
            nodes=C,
            edges=w.astype(np.float64),
            globals=np.array([[hippo.volume / MAX_VOLUME, hippo.y]]),
            senders=e[:, 0],
            receivers=e[:, 1]
        )


def batch_iterate(data: List[Hippocampus], batch_size: int) -> Generator[jraph.GraphsTuple, None, None]:
    return jraph.dynamically_batch(
        iterate(data),
        # Plus one for the extra padding node.
        n_node=batch_size * MAX_NUM_NODES + 1,
        # Times two because we want backwards edges.
        n_edge=batch_size * MAX_NUM_EDGES * 2,
        n_graph=batch_size + 1)


###########################################
# Network
###########################################
class MfdGCN(nn.Module):
    depth: int
    width: int

    @nn.compact
    def __call__(self, G: jraph.GraphsTuple) -> jnp.ndarray:
        # signal domains
        SO = SO3()
        Symp = SPD()

        G_SO = G._replace(nodes=G.nodes[:, :1])
        G_Symp = G._replace(nodes=G.nodes[:, 1:])

        G_SO = G_SO._replace(nodes=jnp.concatenate([G_SO.nodes, ] * self.width, axis=1))
        G_Symp = G_Symp._replace(nodes=jnp.concatenate([G_Symp.nodes, ] * self.width, axis=1))

        G_SO = MfdGcnBlock(SO, [self.width, ] * self.depth)(G_SO)
        G_Symp = MfdGcnBlock(Symp, [self.width, ] * self.depth)(G_Symp)

        # # diffusion layer
        G_SO = FlowLayer(SO)(G_SO)
        G_Symp = FlowLayer(Symp)(G_Symp)

        # invariant layer
        z_SO = MfdInvariant(SO, self.width)(G_SO.nodes[None])[0]
        z_SPD = MfdInvariant(Symp, self.width)(G_Symp.nodes[None])[0]

        z = jax.nn.leaky_relu(jnp.concatenate((z_SO, z_SPD), axis=1))

        # global pooling
        z = jnp.concatenate((max_pooling(G, z), mean_pooling(G, z)), axis=1)
        # add volume information
        z = jnp.append(z, jnp.atleast_2d(G.globals[:, 0]).T, axis=1)

        ### MLP mapping to NUM_CLASSES channels per graph ###
        return MLP((self.width, self.width // 2, NUM_CLASSES))(z)


def main(seed: int):

    # hyperparameters
    learning_rate = 1e-3
    batch_size = 1
    n_epochs = 150

    # initialize network
    network = MfdGCN(4, 16)

    # initialize optimizer
    optimizer = optax.adam(learning_rate)

    # create data
    hippocampi = read_data()

    key = jax.random.key(0)
    params = network.init(key, next(batch_iterate(hippocampi, batch_size)))
    flat_para, _ = jax.flatten_util.ravel_pytree(params)
    print(f"Number of network parameters: {len(flat_para)}")

    # initialize optimizer state
    opt_state = optimizer.init(params)
    state = TrainingState(params, params, opt_state)

    # split data in training and testing sets
    data_train_full, data_test = model_selection.train_test_split(
        hippocampi, test_size=0.2, random_state=seed, stratify=[h.y for h in hippocampi])

    # hold back validation set from the training data
    # data_train, data_validation = model_selection.train_test_split(
    #     data_train_full, test_size=0.25, random_state=seed, stratify=[h.y for h in data_train_full])
    data_train = data_train_full
    data_validation = None

    # evaluate accuracy function
    eval_ = lambda p, g: evaluate(p, g, g.globals[:, 1],
                                  num_classes=NUM_CLASSES,
                                  network=network,
                                  mask=jnp.ones((1,)))

    opt_acc = 0.
    opt_param = state.params
    # training loop
    for i in range(n_epochs):
        # batch-wise training

        for step, batch in enumerate(batch_iterate(data_train, batch_size)):
            mask = jraph.get_graph_padding_mask(batch)

            state = update(state, batch, batch.globals[:, 1], optimizer, network, mask)

        # evaluate accuracy
        train_acc = np.mean([eval_(state.avg_params, g) for g in iterate(data_train)])
        # validation_acc = np.mean([eval_(state.avg_params, g) for g in iterate(data_validation)])
        validation_acc = train_acc
        _test_acc = np.mean([eval_(state.avg_params, g) for g in iterate(data_test)])

        # update optimal parameters (ignore randomly high validation accuracy early in the training)
        if validation_acc > opt_acc and i > 25:
            opt_param = state.avg_params
            # update optimal validation accuracy
            opt_acc = validation_acc

    test_acc = np.mean([eval_(opt_param, g) for g in iterate(data_test)])

    return test_acc


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)

    print(f"\nRunning Manifold GCN on ADNI data using shape-space features with 100 random seeds.")
    results = []
    for s in range(100):
        results.append(main(seed=s))
        print(f"\nSeed: {s}: {results[-1]:.3f}")
        print(f"Running average: {np.average(np.array(results)):.3f}")

    results = np.array(results)
    print(f"\nAverage accuracy {np.mean(results):.3f}, Standard deviation {np.std(results):.3f}")
