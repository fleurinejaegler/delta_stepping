"""
interactive_pathfinder.py
Sélecteur interactif A/B + pathfinding + visualisation.

Usage :
    python interactive_pathfinder.py

1. La carte de danger s'affiche
2. Clique pour placer A (départ, vert)
3. Clique pour placer B (arrivée, rouge)
4. Le chemin optimal est calculé et affiché
5. Clique à nouveau pour recommencer (ou ferme la fenêtre)
"""

import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

# ── CONFIG ────────────────────────────────────────────────────────────────────
GRAPH_PATH  = "graph.npz"
META_PATH   = "meta.npz"
OUTPUT_VTK  = "chemin_optimal.vtk"

# ── CHARGEMENT ────────────────────────────────────────────────────────────────
print("Chargement du graphe...")
g    = np.load(GRAPH_PATH)
meta = np.load(META_PATH, allow_pickle=True)

n_nodes    = int(g['n_nodes'][0])
centers    = meta['centers']
node_score = meta['node_score']
batiment_mask = meta['batiment_mask']
xs         = meta['xs']
ys         = meta['ys']

NX, NY = len(xs), len(ys)

# Graphe
adj = defaultdict(list)
for u, v, w in zip(g['rows'], g['cols'], g['weights']):
    adj[int(u)].append((int(v), float(w)))

tree = cKDTree(centers[:, :2])

# Grilles pour affichage
XX, YY = np.meshgrid(xs, ys)
score_grid    = node_score.copy()
score_grid[batiment_mask] = np.nan
score_grid    = score_grid.reshape(NY, NX)
building_grid = batiment_mask.reshape(NY, NX)

print(f"Prêt — {n_nodes:,} nœuds, {len(g['rows']):,} arêtes")


# ── DELTA-STEPPING ────────────────────────────────────────────────────────────
def calibrer_delta(adj, percentile=25):
    all_w = [w for u in adj for _, w in adj[u] if w > 0 and not np.isinf(w)]
    return float(np.percentile(all_w, percentile))

DELTA = calibrer_delta(adj)


def find_path(source, target):
    INF = float("inf")
    dist = np.full(n_nodes, INF, dtype=np.float64)
    pred = np.full(n_nodes, -1,  dtype=np.int32)
    dist[source] = 0.0

    delta = DELTA
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
        while buckets[i]:
            S = set(buckets[i])
            buckets[i].clear()
            R.update(S)
            for u in S:
                for v, w in adj[u]:
                    if w <= delta:
                        relax(u, v, w)
        for u in R:
            for v, w in adj[u]:
                if w > delta and not np.isinf(w):
                    relax(u, v, w)

    path = []
    if pred[target] >= 0 or target == source:
        cur = target
        while cur != source:
            path.append(cur)
            cur = pred[cur]
        path.append(source)
        path.reverse()

    return dist, path


def export_vtk(path, dist):
    try:
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
        pdata.save(OUTPUT_VTK)
        print(f"  VTK exporté : {OUTPUT_VTK}")
    except Exception as e:
        print(f"  Export VTK échoué : {e}")


# ── VISUALISATION INTERACTIVE ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 9))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

finite_scores = node_score[~batiment_mask & ~np.isinf(node_score)]
vmax = float(np.percentile(finite_scores, 95))
norm = mcolors.Normalize(vmin=0, vmax=vmax)

# Carte de danger
ax.pcolormesh(XX, YY, np.where(building_grid, np.nan, score_grid),
              cmap='YlOrRd', norm=norm, shading='auto', alpha=0.9)
ax.pcolormesh(XX, YY, np.where(building_grid, 1.0, np.nan),
              cmap=mcolors.ListedColormap(['#4a4a6a']),
              shading='auto', alpha=0.85, zorder=2)

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ys[0], ys[-1])
ax.set_aspect('equal')
ax.set_xlabel('X (m)', color='white')
ax.set_ylabel('Y (m)', color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')

title = ax.set_title('Clique pour placer A (départ)', color='white', fontsize=13)

# Éléments dynamiques (mis à jour à chaque interaction)
state = {
    'step': 'A',           # 'A', 'B', ou 'done'
    'markers': [],         # objets matplotlib à nettoyer
    'path_artists': [],
    'text_artist': None,
    'a_xy': None,
    'b_xy': None,
}


def clear_dynamic():
    for obj in state['markers'] + state['path_artists']:
        obj.remove()
    state['markers'].clear()
    state['path_artists'].clear()
    if state['text_artist']:
        state['text_artist'].remove()
        state['text_artist'] = None


def on_click(event):
    if event.inaxes != ax:
        return

    x, y = event.xdata, event.ydata

    # Vérifier que le clic n'est pas dans un bâtiment
    node = int(tree.query([x, y])[1])
    if batiment_mask[node]:
        print(f"  ({x:.1f}, {y:.1f}) est dans un bâtiment — choisis un autre point")
        return

    if state['step'] == 'A':
        clear_dynamic()
        state['a_xy'] = (x, y)
        m = ax.scatter(x, y, c='#00ff88', s=300, marker='*',
                       edgecolors='white', linewidths=1.5, zorder=10)
        state['markers'].append(m)
        title.set_text(f'A = ({x:.1f}, {y:.1f}) — clique pour placer B (arrivée)')
        state['step'] = 'B'
        fig.canvas.draw_idle()

    elif state['step'] == 'B':
        state['b_xy'] = (x, y)
        m = ax.scatter(x, y, c='#ff4466', s=300, marker='*',
                       edgecolors='white', linewidths=1.5, zorder=10)
        state['markers'].append(m)
        title.set_text('Calcul du chemin...')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        # Pathfinding
        ax_val, ay_val = state['a_xy']
        source = int(tree.query([ax_val, ay_val])[1])
        target = int(tree.query([x, y])[1])

        print(f"\n  A = ({ax_val:.1f}, {ay_val:.1f})  →  B = ({x:.1f}, {y:.1f})")
        print(f"  Delta-Stepping...", end=' ', flush=True)

        dist, path = find_path(source, target)

        if path:
            danger = dist[target]
            print(f"OK — {len(path)} nœuds, danger={danger:.3f}")

            # Dessiner le chemin
            path_pts = centers[path]
            if len(path) >= 2:
                segments = [[path_pts[k, :2], path_pts[k+1, :2]]
                            for k in range(len(path)-1)]
                seg_scores = [(node_score[path[k]] + node_score[path[k+1]]) / 2
                              for k in range(len(path)-1)]
                lc = LineCollection(segments, cmap='RdYlGn_r', norm=norm,
                                    linewidth=3, zorder=5, alpha=0.95)
                lc.set_array(np.array(seg_scores))
                ax.add_collection(lc)
                state['path_artists'].append(lc)

            info = (f'Danger cumulé : {danger:.3f}\n'
                    f'Nœuds : {len(path)}\n'
                    f'Moy/nœud : {danger/len(path):.4f}')
            txt = ax.text(0.02, 0.97, info, transform=ax.transAxes,
                         color='white', fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='#2a2a3e',
                                   alpha=0.85, edgecolor='#666'),
                         zorder=11)
            state['text_artist'] = txt

            title.set_text(f'A→B trouvé ! Clique pour recommencer (nouveau A)')

            # Export VTK
            export_vtk(path, dist)
        else:
            print("ÉCHEC — aucun chemin")
            title.set_text('Pas de chemin ! Clique pour recommencer (nouveau A)')

        state['step'] = 'A'
        fig.canvas.draw_idle()


fig.canvas.mpl_connect('button_press_event', on_click)

print("\n╔════════════════════════════════════════════════╗")
print("║  Clique sur la carte pour placer A puis B     ║")
print("║  Le chemin sera calculé automatiquement       ║")
print("║  Clique à nouveau pour recommencer            ║")
print("║  Ferme la fenêtre pour quitter                ║")
print("╚════════════════════════════════════════════════╝\n")

plt.tight_layout()
plt.show()
