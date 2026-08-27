"""
precompute_scores_3d.py
Construction du graphe 3D depuis la carte de dégâts volumique.

Entrée  : damage_map_3d.npz
Sortie  : graph_3d.npz, meta_3d.npz

Grille NX×NY×NZ → graphe bidirectionnel 26-connexe :
  voisins de (i,j,k) : les 26 cases adjacentes
  w(u→v) = score(v) × facteur_distance
           (1 axial, √2 diag 2D, √3 diag 3D)
  bâtiments → score = +inf (infranchissables)

Usage :
    python3 precompute_scores_3d.py
    python3 precompute_scores_3d.py --input damage_map_3d.npz
"""

import argparse
import os
import sys
import numpy as np

# ── ARGUMENTS ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Construction graphe 3D")
parser.add_argument("--input",      default="damage_map_3d.npz")
parser.add_argument("--out_graph",  default="graph_3d.npz")
parser.add_argument("--out_meta",   default="meta_3d.npz")
args = parser.parse_args()

# ── CHARGEMENT CARTE 3D ──────────────────────────────────────────────────────
if not os.path.exists(args.input):
    print(f"ERREUR : {args.input} introuvable.")
    print("Lance d'abord :  python3 generate_3d_map.py")
    sys.exit(1)

print(f"Chargement carte 3D : {args.input}")
gp = np.load(args.input, allow_pickle=True)

xs = gp['xs']                        # (NX,)
ys = gp['ys']                        # (NY,)
zs = gp['zs']                        # (NZ,)
damage_3d    = gp['damage_map_3d']   # (NZ, NY, NX)
building_3d  = gp['building_mask_3d']# (NZ, NY, NX)

NX, NY, NZ = len(xs), len(ys), len(zs)
N = NX * NY * NZ
print(f"  Grille : {NX}×{NY}×{NZ} = {N:,} nœuds")

# ── CONSTRUCTION DES NŒUDS ────────────────────────────────────────────────────
# Indexation : c = k * (NX * NY) + j * NX + i
# Où i ∈ [0,NX), j ∈ [0,NY), k ∈ [0,NZ)
print("Construction des nœuds...")

# Coordonnées 3D de chaque cellule
II, JJ, KK = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing='ij')
# Reshape en (N, 3) — ordre : i varie le plus vite dans le meshgrid 'ij'
# Mais notre indexation c = k*NX*NY + j*NX + i donc on construit en ordre (k,j,i)

centers = np.zeros((N, 3), dtype=np.float64)
node_score    = np.zeros(N, dtype=np.float64)
batiment_mask = np.zeros(N, dtype=bool)

for k in range(NZ):
    for j in range(NY):
        for i in range(NX):
            c = k * (NX * NY) + j * NX + i
            centers[c]       = [xs[i], ys[j], zs[k]]
            node_score[c]    = damage_3d[k, j, i]
            batiment_mask[c] = building_3d[k, j, i]

node_score[batiment_mask] = np.inf

finite = node_score[~np.isinf(node_score)]
print(f"  Bâtiment : {batiment_mask.sum():,} cellules")
print(f"  Scores   : min={finite.min():.4f}  médiane={np.median(finite):.4f}"
      f"  max={finite.max():.4f}")

# ── ARÊTES — 26 VOISINS BIDIRECTIONNEL ───────────────────────────────────────
print("Construction arêtes (26-connexe bidirectionnel)...")

# 26 directions : toutes les combinaisons de {-1,0,1}^3 sauf (0,0,0)
DELTAS = []
for di in (-1, 0, 1):
    for dj in (-1, 0, 1):
        for dk in (-1, 0, 1):
            if di == 0 and dj == 0 and dk == 0:
                continue
            dist = np.sqrt(di**2 + dj**2 + dk**2)  # 1, √2, ou √3
            DELTAS.append((di, dj, dk, dist))

print(f"  {len(DELTAS)} directions (6 axiales + 12 diag-face + 8 diag-volume)")

# Construction vectorisée
all_c = np.arange(N, dtype=np.int64)
all_k = all_c // (NX * NY)
all_j = (all_c % (NX * NY)) // NX
all_i = all_c % NX

rows_list    = []
cols_list    = []
weights_list = []

for di, dj, dk, dist_factor in DELTAS:
    ni = all_i + di
    nj = all_j + dj
    nk = all_k + dk

    valid = (ni >= 0) & (ni < NX) & (nj >= 0) & (nj < NY) & (nk >= 0) & (nk < NZ)

    src = all_c[valid]
    dst = nk[valid] * (NX * NY) + nj[valid] * NX + ni[valid]

    rows_list.append(src.astype(np.int32))
    cols_list.append(dst.astype(np.int32))
    weights_list.append(node_score[dst] * dist_factor)

rows_out    = np.concatenate(rows_list)
cols_out    = np.concatenate(cols_list)
weights_out = np.concatenate(weights_list).astype(np.float64)

n_edges = len(rows_out)
n_inf   = np.isinf(weights_out).sum()
print(f"  {n_edges:,} arêtes  ({n_inf:,} vers bâtiments)")
print(f"  Mémoire estimée : {(n_edges * (4+4+8)) / 1e6:.1f} Mo")

# ── SAUVEGARDE ────────────────────────────────────────────────────────────────
print("Sauvegarde...")

np.savez(args.out_graph,
         rows=rows_out,
         cols=cols_out,
         weights=weights_out,
         n_nodes=np.array([N], dtype=np.int64))

np.savez(args.out_meta,
         centers=centers,
         node_score=node_score,
         batiment_mask=batiment_mask,
         xs=xs, ys=ys, zs=zs,
         NX=np.array([NX]),
         NY=np.array([NY]),
         NZ=np.array([NZ]))

g_size = os.path.getsize(args.out_graph) / 1e6 if os.path.exists(args.out_graph) else 0
m_size = os.path.getsize(args.out_meta)  / 1e6 if os.path.exists(args.out_meta)  else 0

print(f"\nSauvegardé :")
print(f"  {args.out_graph}  ({g_size:.1f} Mo) — {N:,} nœuds, {n_edges:,} arêtes")
print(f"  {args.out_meta}   ({m_size:.1f} Mo)")
