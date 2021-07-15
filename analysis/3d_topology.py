import networkx as netx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modules.topologies import SmallWorldTopology
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from collections import Counter

if __name__ == "__main__":
    topology = SmallWorldTopology(
        SmallWorldTopology.Configuration(
            minicolumn_shape=(4, 4, 4),
            macrocolumn_shape=(2, 2, 2),
            minicolumn_spacing=1460,
            p_max=0.11,
            intracolumnar_sparseness=635,
            neuron_spacing=40,
            inhibitory_init_weight_range=(0.1, 0.3),
            excitatory_init_weight_range=(0.2, 0.5),
        )
    )
    nodes = topology.nodes(data="position")
    positions = np.array([p for [_, p] in nodes]).T

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")

    excitatory_conn = []
    inhibitory_conn = []
    for u, v, w in topology.edges(data="weight"):
        l = excitatory_conn if w > 0 else inhibitory_conn
        l.append(np.array((positions[:, u], positions[:, v])))

    excitatory_lines = Line3DCollection(excitatory_conn, colors=[(0, 1, 0, 0.1)] * len(excitatory_conn))
    inhibitory_lines = Line3DCollection(inhibitory_conn, colors=[(1, 0, 0, 0.1)] * len(inhibitory_conn))
    ax.add_collection(excitatory_lines)
    ax.add_collection(inhibitory_lines)

    ax.scatter(positions[0], positions[1], positions[2])
    ax.axis("off")

    if display_numbers := False:
        _, macro_indices, macro_inverse = np.unique(positions % 40, axis=1, return_index=True, return_inverse=True)
        macro_positions = positions[:, macro_indices]
        macro_counter = Counter()
        for u, v in topology.edges:
            macro_counter[tuple(sorted((macro_inverse[u], macro_inverse[v])))] += 1

        macro_ids = np.unique(macro_inverse)
        for u, pos_u in zip(macro_ids, macro_positions.T):
            for v, pos_v in zip(macro_ids, macro_positions.T):
                if macro_counter[u, v] == 0:
                    continue

                mean_pos = np.mean(positions[:, (macro_inverse == u) | (macro_inverse == v)], axis=1) + 0.05 * np.mean(
                    positions[:, (macro_inverse == u)], axis=1
                )
                mean_pos[2] += 5

                ax.text(*mean_pos, "%i" % macro_counter[u, v], fontsize=20, zorder=20)

    plt.tight_layout()
    fig.savefig("3d_topology.pdf", bbox_inches="tight")

    plt.show()
