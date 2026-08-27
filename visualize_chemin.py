"""
visualize_chemin.py
Visualisation du chemin optimal sur la carte de danger GP.

Entrée  : chemin_optimal.vtk + meta.npz
Sortie  : visualisation_chemin.png

Panneau gauche : carte des scores de danger (heatmap)
Panneau droit  : chemin optimal coloré par score local

Ne nécessite pas les fichiers de scène (contours_frt.data, VTK, etc.)
Les bâtiments sont dessinés depuis le building_mask de meta.npz.
"""

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

# ── CONFIG ────────────────────────────────────────────────────────────────────
META_PATH   = "meta.npz"
CHEMIN_PATH = "chemin_optimal.vtk"
OUTPUT_IMG  = "visualisation_chemin.png"

A_XY = np.array([20.0, 20.0])
B_XY = np.array([80.0, 80.0])

# ── CHARGEMENT ────────────────────────────────────────────────────────────────
print("Chargement...")
meta          = np.load(META_PATH, allow_pickle=True)
centers       = meta['centers']
node_score    = meta['node_score']
batiment_mask = meta['batiment_mask']
xs            = meta['xs']
ys            = meta['ys']

chemin      = pv.read(CHEMIN_PATH)
path_pts    = np.array(chemin.points)
path_scores = np.array(chemin['score_local'])
path_dist   = np.array(chemin['dist_cumulee'])

NX, NY = len(xs), len(ys)

# Reconstruire la grille pour pcolormesh
XX, YY = np.meshgrid(xs, ys)
score_grid    = node_score.copy()
score_grid[batiment_mask] = np.nan
score_grid    = score_grid.reshape(NY, NX)
building_grid = batiment_mask.reshape(NY, NX)

# ── VISUALISATION ─────────────────────────────────────────────────────────────
print("Génération figure...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('#1a1a2e')

finite_scores = node_score[~batiment_mask & ~np.isinf(node_score)]
vmax = float(np.percentile(finite_scores, 95))
norm = mcolors.Normalize(vmin=0, vmax=vmax)

for ax in axes:
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')


def draw_base(ax, alpha=0.8):
    """Heatmap scores + bâtiments (depuis le masque)."""
    im = ax.pcolormesh(XX, YY, np.where(building_grid, np.nan, score_grid),
                       cmap='YlOrRd', norm=norm, shading='auto', alpha=alpha)
    # Dessiner les bâtiments en gris
    ax.pcolormesh(XX, YY, np.where(building_grid, 1.0, np.nan),
                  cmap=mcolors.ListedColormap(['#4a4a6a']),
                  shading='auto', alpha=0.85, zorder=2)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_aspect('equal')
    return im


# ── Panneau gauche : carte complète ───────────────────────────────────────────
ax = axes[0]
ax.set_title('Carte des scores de danger (GP)', color='white', fontsize=13, pad=10)
im = draw_base(ax)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score de danger', color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
ax.scatter(*A_XY, c='#00ff88', s=250, marker='*', zorder=10, label='A (départ)')
ax.scatter(*B_XY, c='#ff4466', s=250, marker='*', zorder=10, label='B (arrivée)')
ax.legend(facecolor='#2a2a3e', labelcolor='white', fontsize=9)
ax.set_xlabel('X (m)', color='white')
ax.set_ylabel('Y (m)', color='white')

# ── Panneau droit : chemin optimal ────────────────────────────────────────────
ax = axes[1]
ax.set_title('Chemin optimal A → B (moindre danger)', color='white',
             fontsize=13, pad=10)
draw_base(ax, alpha=0.25)

# Segments colorés par score local
n = len(path_pts)
if n >= 2:
    segments   = np.array([[path_pts[k, :2], path_pts[k+1, :2]]
                            for k in range(n - 1)])
    seg_scores = (path_scores[:-1] + path_scores[1:]) / 2
    lc = LineCollection(segments, cmap='RdYlGn_r',
                        norm=norm, linewidth=2.5, zorder=5, alpha=0.95)
    lc.set_array(seg_scores)
    ax.add_collection(lc)
    cbar2 = plt.colorbar(lc, ax=ax)
    cbar2.set_label('Score sur le chemin', color='white')
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white')
    ax.scatter(path_pts[:, 0], path_pts[:, 1],
               c='white', s=8, zorder=6, alpha=0.5)

danger_total = float(path_dist[-1]) if len(path_dist) else 0.0
ax.text(0.02, 0.97,
        f'Danger cumulé : {danger_total:.3f}\n'
        f'Nœuds         : {n}\n'
        f'Moy/nœud      : {danger_total/max(n,1):.4f}',
        transform=ax.transAxes, color='white', fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#2a2a3e',
                  alpha=0.8, edgecolor='#666'))

ax.scatter(*A_XY, c='#00ff88', s=250, marker='*', zorder=10, label='A (départ)')
ax.scatter(*B_XY, c='#ff4466', s=250, marker='*', zorder=10, label='B (arrivée)')
ax.legend(facecolor='#2a2a3e', labelcolor='white', fontsize=9)
ax.set_xlabel('X (m)', color='white')
ax.set_ylabel('Y (m)', color='white')

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"Image sauvegardée : {OUTPUT_IMG}")
