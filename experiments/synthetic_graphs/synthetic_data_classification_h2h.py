import sys

import jax
import jax.numpy as jnp
import jraph

from typing import Callable

import numpy as np
import flax.linen as nn
import optax

from morphomatics.nn.euclidean_layers import MLP
from morphomatics.graph.operators import mean_pooling
from morphomatics.manifold.util import multiskew
from morphomatics.manifold import HyperbolicSpace

from synthetic_data_classification import generate_data, train, NUM_NODES, NUM_CLASSES

EPS = 1e-6


def lorentz_to_klein(x):
    return x[:-1] / x[-1]


def klein_to_lorentz(k: jnp.array, M: HyperbolicSpace):
    x = jnp.array((*k, 1))
    x = x / jnp.sqrt(1 - k.T @ k)
    return M.project_to_manifold(x)


def lorentz_to_poincare(x: jnp.array):
    return x[:-1] / (x[-1] + 1)


def poincare_to_lorentz(b: jnp.array, M: HyperbolicSpace):
    x = jnp.array((*2*b, 1 + b.T @ b))
    x = x / (1 - b.T @ b)
    return M.project_to_manifold(x)


def random_rotations(key, shape):
    S = jax.random.normal(key, shape)
    S = multiskew(S)
    return jax.vmap(jax.linalg.expm)(S)


def multi_triu(X, d):
    """Turn an array of size n x m, where m = (d - 1) * d / 2 for some positive integer d, into an array of size
    n x d x d.
    Each slice is an upper triangular matrix."""

    def single_triu(x):
        U = jnp.zeros((d, d))
        return U.at[jnp.triu_indices(d, 1)].set(x)

    return jax.vmap(single_triu)(X)


class H2HNet(nn.Module):
    """Hyperbolic-to-Hyperbolic network as introduced in

        Dai, J., Wu, Y., Gao, Z., & Jia, Y. (2021). A hyperbolic-to-hyperbolic graph convolutional network.
        In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 154-163).

    :param num_layers: number of layers in the network
    :param num_centroids: number of points (weighted means) to which distances are employed in the invariant layer
    :param dim: dimension of the hyperbolic space
    :param s_init: function used to initialize the orthogonal weight matrices
    :param centroid_init: function used to initialize the centroids

    """
    num_layers: int
    num_centroids: int
    dim: int = NUM_NODES
    s_init: Callable = nn.initializers.normal(stddev=1.)
    centroid_init: Callable = nn.initializers.normal(stddev=0.5)

    @nn.compact
    def __call__(self, G: jraph.GraphsTuple) -> jnp.ndarray:
        M = HyperbolicSpace((self.dim + 1,))

        to_tan = nn.Dense(self.dim, use_bias=False)
        z = jax.vmap(to_tan)(G.nodes)
        # interpret z as an element of the tangent space at p
        z = jnp.concatenate((z, jnp.zeros((len(z), 1))), axis=1)

        h = jax.vmap(lambda v: M.project_to_manifold(M.connec.exp(M.pole(), v)))(z)

        # dimension of SO(NUM_NODES)
        d = int((self.dim - 1) * self.dim / 2)

        # learnable parameters are the coordinates of the rotations in the standard chart of SO(self.dim), i.e.,
        # the entries of the inverses in the Lie algebra under the group logarithm
        s = self.param("skew_vals", self.s_init, (self.num_layers, d))
        S = multi_triu(s, self.dim)
        S = S - jnp.einsum("...ij->...ji", S)

        # map skew-symmetrix matrices with the Lie group exponential to SO(NUM_NODES)
        R = jax.vmap(jax.scipy.linalg.expm)(S)

        # add additional dimension to accommodate the Lorentz model
        W = jnp.zeros((self.num_layers, self.dim + 1, self.dim + 1))
        W = W.at[:, :-1, :-1].set(R)
        W = W.at[:, -1, -1].set(1)

        def node_aggregation(h_l):
            h_l = jax.vmap(lorentz_to_klein)(h_l)
            gam_l = jax.vmap(lambda v: 1/jnp.sqrt(1 - v.T @ v))(h_l)
            h_gam_l = h_l * gam_l[:, None]

            # sum nodes do not have outgoing edges
            def cond_fun_sg(i: int, g):
                return i+1, jax.lax.cond(g == 0, lambda _: 1., lambda x: x, g)

            sum_gam = jax.ops.segment_sum(gam_l[G.receivers], G.senders, num_segments=len(G.nodes))
            _, sum_gam = jax.lax.scan(cond_fun_sg, 0, sum_gam)

            # sum nodes do not have outgoing edges
            def cond_fun_h(i: int, m):
                return i+1, jax.lax.cond(jnp.linalg.norm(m) > 0, lambda x: x, lambda _: h_l[i], m)

            m_l = jax.ops.segment_sum(h_gam_l[G.receivers], G.senders, num_segments=len(G.nodes)) / sum_gam[:, None]
            _, m_l = jax.lax.scan(cond_fun_h, 0, m_l)
            m_l = jax.vmap(klein_to_lorentz)(m_l, M)  # change back to m_l

            m_l = nn.relu(jax.vmap(lorentz_to_poincare)(m_l))
            m_l = jax.vmap(poincare_to_lorentz)(m_l, M)

            return m_l

        def h2h_layer(i, h_l):
            h_l = jax.vmap(lambda x: M.project_to_manifold(W[i] @ x))(h_l)
            return node_aggregation(h_l)

        h = jax.lax.fori_loop(0, self.num_layers, h2h_layer, h)
        h = jax.vmap(M.project_to_manifold)(h)

        c = self.param("centroids", self.centroid_init, (self.num_centroids, self.dim))
        c = jnp.concatenate((c, jnp.zeros((self.num_centroids, 1))), axis=1)
        c = jax.vmap(M.connec.exp, in_axes=(None, 0))(M.pole(), c)

        d = jax.vmap(jax.vmap(M.metric.dist, in_axes=(None, 0)), in_axes=(0, None))(h, c)
        z = mean_pooling(G, d)

        # mapping to NUM_CLASSES channels per graph
        f = MLP((NUM_CLASSES,))
        return f(z)


def main(graph_num: int,
         dim: int,
         batch_size: int,
         n_val: int,
         n_test: int,
         n_epoch: int,
         seed: jnp.ndarray) -> float:

    data = generate_data(graph_num, "hyperbolic", "degree", batch_size, n_val, n_test, seed)

    network = H2HNet(4, 15, dim)  # number of layers and channels are the best parameters according to hyperparameter search
    optimizer = optax.adam(1e-3)

    test_f1, _, _ = train(network, optimizer, data, n_epoch, jax.random.key(42))

    return test_f1


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    num_graphs = int(sys.argv[1])
    hyperbolic_dimension = int(sys.argv[2])
    n_repeats = int(sys.argv[3])

    assert num_graphs  > 0

    size_batches= 3
    num_val = int(num_graphs / 6)
    num_test = int(num_graphs / 6)
    num_epochs = 60

    results = []

    print(f"\nStart training on {3 * num_graphs} graphs.")
    for s in range(n_repeats):
        results.append(main(graph_num=num_graphs,
                            dim=hyperbolic_dimension,
                            batch_size=size_batches,
                            n_val=num_val,
                            n_test=num_test,
                            n_epoch=num_epochs,
                            seed=jax.random.key(s)))

        print(f"\nF1 score seed {s}: {results[-1]:.3f}")
        print(f"Running average: {np.mean(results):.3f}")

    print(f"\nAverage F1 score: {np.mean(results):.3f}, Standard Deviation: {np.std(results):.3f}")



