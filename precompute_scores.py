"""
precompute_scores.py
Construction du graphe depuis la carte GP (damage_map.npz).

Entrée  : damage_map.npz  (exporté par gp_damage.py export)
Sortie  : graph.npz, meta.npz

Grille 100×100 → graphe bidirectionnel 8-connexe :
  voisins de (i,j) : les 8 cases adjacentes
  w(u→v) = score(v)  ×  facteur_distance
           (√2 pour les diagonales, 1 pour les axes)
  bâtiments → score = +inf (infranchissables)
"""

import os
import sys
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
GP_MAP_PATH = "damage_map.npz"
OUT_GRAPH   = "graph.npz"
OUT_META    = "meta.npz"

# ── CHARGEMENT CARTE GP ──────────────────────────────────────────────────────
if not os.path.exists(GP_MAP_PATH):
    print(f"ERREUR : {GP_MAP_PATH} introuvable.")
    print("Lance d'abord :  python gp_damage.py export")
    sys.exit(1)

print(f"Chargement carte GP : {GP_MAP_PATH}")
gp = np.load(GP_MAP_PATH)

xs            = gp['xs']             # (NX,)
ys            = gp['ys']             # (NY,)
damage_map    = gp['damage_map']     # (NY, NX)
building_mask = gp['building_mask']  # (NY, NX)

NX, NY = len(xs), len(ys)
N      = NX * NY
print(f"  Grille : {NX}×{NY} = {N} nœuds")

# ── CONSTRUCTION DES NŒUDS ────────────────────────────────────────────────────
# Indexation : c = j * NX + i
XX, YY = np.meshgrid(xs, ys)
centers = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(N)])

node_score    = damage_map.ravel().copy().astype(np.float64)
batiment_mask = building_mask.ravel().copy()
node_score[batiment_mask] = np.inf

# Mappings grille ↔ nœud
cell_to_ij = np.column_stack([
    np.tile(np.arange(NX), NY),
    np.repeat(np.arange(NY), NX),
]).astype(np.int32)

ij_to_cell = np.arange(N, dtype=np.int32).reshape(NY, NX).T.copy()

finite = node_score[~np.isinf(node_score)]
print(f"  Bâtiment : {batiment_mask.sum():,} cellules")
print(f"  Scores   : min={finite.min():.4f}  médiane={np.median(finite):.4f}"
      f"  max={finite.max():.4f}")

# ── ARÊTES — 8 VOISINS BIDIRECTIONNEL ────────────────────────────────────────
print("Construction arêtes (8-connexe bidirectionnel)...")

# 8 directions + facteur distance (√2 pour diagonales)
DELTAS = [
    ( 1,  0, 1.0),    # droite
    (-1,  0, 1.0),    # gauche
    ( 0,  1, 1.0),    # haut
    ( 0, -1, 1.0),    # bas
    ( 1,  1, 1.414),  # diag ↗
    ( 1, -1, 1.414),  # diag ↘
    (-1,  1, 1.414),  # diag ↖
    (-1, -1, 1.414),  # diag ↙
]

all_i = cell_to_ij[:, 0]
all_j = cell_to_ij[:, 1]

rows_list, cols_list, weights_list = [], [], []

for di, dj, dist_factor in DELTAS:
    ni = all_i + di
    nj = all_j + dj
    valid = (ni >= 0) & (ni < NX) & (nj >= 0) & (nj < NY)

    src = np.arange(N)[valid]
    dst = nj[valid] * NX + ni[valid]

    rows_list.append(src)
    cols_list.append(dst)
    weights_list.append(node_score[dst] * dist_factor)

rows_out    = np.concatenate(rows_list).astype(np.int32)
cols_out    = np.concatenate(cols_list).astype(np.int32)
weights_out = np.concatenate(weights_list).astype(np.float64)

print(f"  {len(rows_out):,} arêtes  "
      f"({np.isinf(weights_out).sum():,} vers bâtiments)")

# ── SAUVEGARDE ────────────────────────────────────────────────────────────────
np.savez(OUT_GRAPH,
         rows=rows_out, cols=cols_out, weights=weights_out,
         n_nodes=np.array([N]))

np.savez(OUT_META,
         centers=centers,
         node_score=node_score,
         batiment_mask=batiment_mask,
         cell_to_ij=cell_to_ij,
         ij_to_cell=ij_to_cell,
         xs=xs, ys=ys)

print(f"\nSauvegardé : {OUT_GRAPH}  ({N} nœuds, {len(rows_out):,} arêtes)")
print(f"           : {OUT_META}")
