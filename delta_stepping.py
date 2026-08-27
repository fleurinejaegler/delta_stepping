"""
delta_stepping.py
Delta-Stepping séquentiel — graphe de dégâts urbains.

Entrée  : graph.npz + meta.npz
Sortie  : chemin_optimal.vtk

Usage :
    python delta_stepping.py                  # A=(10,50) B=(92,50) par défaut
    python delta_stepping.py 15 30 85 70      # A=(15,30) B=(85,70)
"""

import sys
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree

# ── CONFIG ────────────────────────────────────────────────────────────────────
GRAPH_PATH = "graph.npz"
META_PATH  = "meta.npz"
OUTPUT_VTK = "chemin_optimal.vtk"

# Coordonnées par défaut (modifiables en ligne de commande)
DEFAULT_A = (20.0, 20.0)
DEFAULT_B = (80.0, 80.0)


# ── CHARGEMENT ────────────────────────────────────────────────────────────────
def load_graph(graph_path, meta_path):
    g    = np.load(graph_path)
    meta = np.load(meta_path, allow_pickle=True)

    n_nodes    = int(g['n_nodes'][0])
    rows       = g['rows']
    cols       = g['cols']
    weights    = g['weights']
    centers    = meta['centers']
    node_score = meta['node_score']

    adj = defaultdict(list)
    for u, v, w in zip(rows, cols, weights):
        adj[int(u)].append((int(v), float(w)))

    print(f"Graphe : {n_nodes:,} nœuds  {len(rows):,} arêtes")
    return adj, n_nodes, centers, node_score


def calibrer_delta(adj, percentile=25):
    all_w = [w for u in adj for _, w in adj[u] if w > 0 and not np.isinf(w)]
    all_w = np.array(all_w)
    delta = float(np.percentile(all_w, percentile))
    print(f"Delta (p{percentile}) : {delta:.4f}")
    return delta


# ── DELTA-STEPPING ────────────────────────────────────────────────────────────
def delta_stepping(adj, n_nodes, source, target, delta):
    INF = float("inf")
    dist = np.full(n_nodes, INF, dtype=np.float64)
    pred = np.full(n_nodes, -1,  dtype=np.int32)
    dist[source] = 0.0

    def bucket_idx(d):
        return int(d / delta)

    buckets = defaultdict(set)
    buckets[0].add(source)

    def relax(u, v, w):
        if np.isinf(w):
            return
        new_d = dist[u] + w
        if new_d < dist[v]:
            old_d = dist[v]
            dist[v] = new_d
            pred[v] = u
            if old_d < INF:
                buckets[bucket_idx(old_d)].discard(v)
            buckets[bucket_idx(new_d)].add(v)

    while True:
        active = [k for k, b in buckets.items() if b]
        if not active:
            break
        i = min(active)

        if dist[target] < INF and i > bucket_idx(dist[target]):
            break

        R = set()

        # Phase légère
        while buckets[i]:
            S = set(buckets[i])
            buckets[i].clear()
            R.update(S)
            for u in S:
                for v, w in adj[u]:
                    if w <= delta:
                        relax(u, v, w)

        # Phase lourde
        for u in R:
            for v, w in adj[u]:
                if w > delta and not np.isinf(w):
                    relax(u, v, w)

    # Reconstruction chemin
    path = []
    if pred[target] >= 0 or target == source:
        cur = target
        while cur != source:
            path.append(cur)
            cur = pred[cur]
        path.append(source)
        path.reverse()

    return dist, pred, path


def export_chemin_vtk(path, centers, dist, node_score, output):
    import pyvista as pv
    n   = len(path)
    pts = centers[path]

    lines = np.hstack([
        np.full((n - 1, 1), 2, dtype=np.intp),
        np.column_stack([np.arange(n - 1), np.arange(1, n)])
    ]).ravel()

    pdata = pv.PolyData(pts)
    pdata.lines           = lines
    pdata["dist_cumulee"] = dist[path]
    pdata["score_local"]  = node_score[path]
    pdata.save(output)
    print(f"Chemin exporté : {output}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Parse arguments optionnels : python delta_stepping.py Ax Ay Bx By
    if len(sys.argv) == 5:
        A_XY = np.array([float(sys.argv[1]), float(sys.argv[2])])
        B_XY = np.array([float(sys.argv[3]), float(sys.argv[4])])
    else:
        A_XY = np.array(DEFAULT_A)
        B_XY = np.array(DEFAULT_B)

    adj, n_nodes, centers, node_score = load_graph(GRAPH_PATH, META_PATH)

    tree   = cKDTree(centers[:, :2])
    source = int(tree.query(A_XY)[1])
    target = int(tree.query(B_XY)[1])

    print(f"Source A : nœud {source}  ({centers[source, 0]:.1f}, {centers[source, 1]:.1f})")
    print(f"Cible  B : nœud {target}  ({centers[target, 0]:.1f}, {centers[target, 1]:.1f})")

    delta = calibrer_delta(adj)
    print(f"\nDelta-Stepping (delta={delta:.4f})...")

    dist, pred, path = delta_stepping(adj, n_nodes, source, target, delta)

    if path:
        print(f"\nDanger cumulé total  : {dist[target]:.4f}")
        print(f"Longueur du chemin   : {len(path)} nœuds")
        print(f"Danger moyen/nœud    : {dist[target]/len(path):.4f}")
        export_chemin_vtk(path, centers, dist, node_score, OUTPUT_VTK)
    else:
        print("\nAucun chemin trouvé entre A et B.")
        print("Les bâtiments bloquent peut-être le passage.")
        print("Essaie d'autres coordonnées avec : python delta_stepping.py Ax Ay Bx By")
