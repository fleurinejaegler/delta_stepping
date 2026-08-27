"""
interpolate_terrain.py
Interpole le maillage triangulaire de terrain (output-post.vtk)
sur la grille régulière de damage_map_3d.npz.

Entrée  : output-post.vtk  (maillage triangulaire de surface)
          damage_map_3d.npz (carte de dégâts 3D existante)
Sortie  : damage_map_3d.npz (mis à jour avec terrain_elevation)
          terrain.npz        (carte d'élévation seule)

Les Cell IDs du VTK sont traités comme identifiants de bâtiments :
  - Les triangles avec un ID de bâtiment marquent les cellules de la grille
    comme bâtiment sur toute la hauteur du bâtiment.

Usage :
    python3 interpolate_terrain.py
    python3 interpolate_terrain.py --vtk output-post.vtk --damage damage_map_3d.npz
"""

import argparse
import os
import sys
import struct
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# ── ARGUMENTS ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Interpolation terrain VTK → grille régulière")
parser.add_argument("--vtk",         default="output-post.vtk",
                    help="Maillage triangulaire du terrain")
parser.add_argument("--damage",      default="damage_map_3d.npz",
                    help="Carte de dégâts 3D existante")
parser.add_argument("--terrain_out", default="terrain.npz",
                    help="Carte d'élévation seule")
parser.add_argument("--h_bat",       type=float, default=None,
                    help="Hauteur des bâtiments en m (si None, utilise celle de damage_map_3d.npz)")
args = parser.parse_args()

# ── LECTURE DU VTK BINAIRE ────────────────────────────────────────────────────
print(f"Lecture du maillage terrain : {args.vtk}")

with open(args.vtk, 'rb') as f:
    # Header (5 lignes ASCII)
    header1 = f.readline().decode().strip()
    header2 = f.readline().decode().strip()
    encoding = f.readline().decode().strip()    # BINARY ou ASCII
    dataset  = f.readline().decode().strip()    # DATASET POLYDATA
    points_line = f.readline().decode().strip() # POINTS N float

    parts = points_line.split()
    n_points = int(parts[1])
    dtype_str = parts[2]

    print(f"  {n_points} points, format={encoding}, dtype={dtype_str}")

    # Points (big-endian)
    if dtype_str == 'float':
        raw = f.read(n_points * 3 * 4)
        points = np.frombuffer(raw, dtype='>f4').reshape(n_points, 3).astype(np.float64)
    else:
        raw = f.read(n_points * 3 * 8)
        points = np.frombuffer(raw, dtype='>f8').reshape(n_points, 3).copy()
    f.read(1)  # newline

    print(f"  X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
    print(f"  Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
    print(f"  Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

    # POLYGONS
    poly_line = f.readline().decode().strip()
    parts = poly_line.split()
    n_polys = int(parts[1])
    n_total = int(parts[2])
    print(f"  {n_polys} polygones (triangles)")

    # OFFSETS vtktypeint64
    f.readline()  # OFFSETS line
    raw_off = f.read(n_polys * 8)
    offsets = np.frombuffer(raw_off, dtype='>i8').copy()
    f.read(1)

    # CONNECTIVITY vtktypeint64
    f.readline()  # CONNECTIVITY line
    raw_conn = f.read(n_total * 8)
    connectivity = np.frombuffer(raw_conn, dtype='>i8').copy()
    f.read(1)

    # Triangles (3 sommets par polygone)
    # Le nombre de triangles = n_polys - 1 ou n_polys selon le format
    # offsets donne les positions de début de chaque polygone
    n_triangles = n_polys
    if len(connectivity) >= n_triangles * 3:
        triangles = connectivity[:n_triangles * 3].reshape(-1, 3)
    else:
        # Utiliser les offsets pour parser
        polys = []
        for t in range(n_triangles):
            start = int(offsets[t - 1]) if t > 0 else 0
            end = int(offsets[t])
            polys.append(connectivity[start:end])
        # Garder uniquement les triangles (3 sommets)
        triangles = np.array([p for p in polys if len(p) == 3], dtype=np.int64)

    print(f"  Triangles parsés : {triangles.shape}")

    # CELL_DATA — identifiants bâtiments
    cell_line = f.readline().decode().strip()
    n_cells_data = int(cell_line.split()[1])

    scalars_line = f.readline().decode().strip()  # SCALARS ID int
    lookup_line  = f.readline().decode().strip()   # LOOKUP_TABLE default

    raw_ids = f.read(n_cells_data * 4)
    cell_ids = np.frombuffer(raw_ids, dtype='>i4').copy()

    print(f"  Cell IDs : {n_cells_data} valeurs, {len(np.unique(cell_ids))} uniques")
    print(f"  ID range : [{cell_ids.min()}, {cell_ids.max()}]")

# ── CHARGEMENT GRILLE EXISTANTE ───────────────────────────────────────────────
print(f"\nChargement grille : {args.damage}")
d = np.load(args.damage, allow_pickle=True)

xs = d['xs']
ys = d['ys']
zs = d['zs']
NX, NY, NZ = len(xs), len(ys), len(zs)

h_bat = float(d['h_bat'][0]) if args.h_bat is None else args.h_bat

print(f"  Grille : {NX}×{NY}×{NZ}")
print(f"  X: [{xs[0]:.2f}, {xs[-1]:.2f}]")
print(f"  Y: [{ys[0]:.2f}, {ys[-1]:.2f}]")
print(f"  Z: [{zs[0]:.2f}, {zs[-1]:.2f}]")
print(f"  Hauteur bâtiments : {h_bat:.1f}m")

# ── INTERPOLATION DU TERRAIN ─────────────────────────────────────────────────
print("\nInterpolation de l'élévation sur la grille régulière...")

# Coordonnées XY des points du maillage
pts_xy = points[:, :2]
pts_z  = points[:, 2]

# Interpolation linéaire (+ nearest pour les points hors du maillage)
interp_linear  = LinearNDInterpolator(pts_xy, pts_z)
interp_nearest = NearestNDInterpolator(pts_xy, pts_z)

XX, YY = np.meshgrid(xs, ys)  # (NY, NX)
grid_xy = np.column_stack([XX.ravel(), YY.ravel()])

terrain_flat = interp_linear(grid_xy)

# Remplir les NaN (hors de l'enveloppe convexe) avec nearest
nan_mask = np.isnan(terrain_flat)
if nan_mask.any():
    terrain_flat[nan_mask] = interp_nearest(grid_xy[nan_mask])
    print(f"  {nan_mask.sum()} points hors maillage remplis par nearest")

terrain = terrain_flat.reshape(NY, NX)

print(f"  Terrain interpolé : min={terrain.min():.2f}m  max={terrain.max():.2f}m  moy={terrain.mean():.2f}m")

# ── IDENTIFICATION DES BÂTIMENTS SUR LA GRILLE ───────────────────────────────
print("\nIdentification des bâtiments sur la grille...")

# Pour chaque triangle, calculer son centroïde et vérifier
# si son Cell ID correspond à un bâtiment
# On considère que les Cell IDs sont des identifiants de bâtiments
# → chaque ID unique correspond à un bâtiment distinct

# Centroïdes des triangles
tri_centroids = np.zeros((len(triangles), 2), dtype=np.float64)
for t in range(len(triangles)):
    verts = points[triangles[t]]
    tri_centroids[t] = verts[:, :2].mean(axis=0)

# Pour chaque cellule de la grille, trouver le triangle le plus proche
# et récupérer son Cell ID
from scipy.spatial import cKDTree

tree_tri = cKDTree(tri_centroids)

# Centroïdes de la grille
grid_centroids = np.column_stack([XX.ravel(), YY.ravel()])
_, nearest_tri = tree_tri.query(grid_centroids)

# Mapper les Cell IDs sur la grille
# Attention : n_cells_data peut être n_triangles - 1
grid_cell_ids = np.zeros(NX * NY, dtype=np.int32)
for idx in range(len(grid_centroids)):
    tri_idx = nearest_tri[idx]
    if tri_idx < len(cell_ids):
        grid_cell_ids[idx] = cell_ids[tri_idx]
    else:
        grid_cell_ids[idx] = -1

grid_cell_ids = grid_cell_ids.reshape(NY, NX)

# Identifier les bâtiments : on garde l'info de la carte 2D existante
# et on l'enrichit avec les Cell IDs du terrain
building_2d_original = d['building_mask_2d']  # (NY, NX)

print(f"  Bâtiments originaux (masque 2D) : {building_2d_original.sum():,} cellules")
print(f"  Cell IDs sur la grille : {len(np.unique(grid_cell_ids))} uniques")

# ── MISE À JOUR DU MASQUE BÂTIMENT 3D ────────────────────────────────────────
print("\nRecalcul du masque bâtiment 3D avec le terrain...")

damage_3d   = d['damage_map_3d'].copy()
building_3d = d['building_mask_3d'].copy()

# Recalculer : bâtiments entre terrain(x,y) et terrain(x,y) + h_bat
building_3d[:] = False
for k in range(NZ):
    z = zs[k]
    above_ground = z >= terrain
    below_roof   = z <= terrain + h_bat
    building_3d[k, :, :] = building_2d_original & above_ground & below_roof

print(f"  Bâtiments 3D : {building_3d.sum():,} cellules")

# ── SAUVEGARDE ────────────────────────────────────────────────────────────────
print("\nSauvegarde...")

all_data = dict(d)
all_data['terrain_elevation'] = terrain           # (NY, NX)
all_data['building_mask_3d']  = building_3d       # mis à jour
all_data['grid_cell_ids']     = grid_cell_ids     # (NY, NX) IDs bâtiments
d.close()

np.savez(args.damage, **all_data)
print(f"  Mis à jour : {args.damage}")
print(f"    Nouvelles clés : terrain_elevation, grid_cell_ids")

np.savez(args.terrain_out,
         xs=xs, ys=ys,
         terrain_elevation=terrain,
         grid_cell_ids=grid_cell_ids,
         # Garder aussi le maillage original pour référence
         mesh_points=points,
         mesh_triangles=triangles,
         mesh_cell_ids=cell_ids)
print(f"  Terrain seul : {args.terrain_out}")

print(f"\nRésumé :")
print(f"  Grille : {NX}×{NY}×{NZ}")
print(f"  Terrain : z ∈ [{terrain.min():.2f}, {terrain.max():.2f}]m")
print(f"  Bâtiments 3D : {building_3d.sum():,} cellules")
print(f"\nProchaine étape :")
print(f"  python3 precompute_scores_3d.py")
print(f"  mpirun -np 4 python3 delta_stepping_mpi_3d.py --z_above_ground 1.0")
