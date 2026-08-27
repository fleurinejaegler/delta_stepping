"""
generate_3d_map.py
Génère une carte de dégâts 3D synthétique à partir de la carte 2D existante.

Entrée  : damage_map.npz  (carte 2D : xs, ys, damage_map, building_mask)
Sortie  : damage_map_3d.npz

Modèle :
  - Le danger au sol (z=0) est identique à la carte 2D
  - Le danger décroît avec l'altitude : score_3d(x,y,z) = score_2d(x,y) * exp(-z / H)
  - Les bâtiments sont extrudés verticalement jusqu'à une hauteur donnée
  - Au-dessus des toits, le danger reprend (mais atténué)

Usage :
    python generate_3d_map.py
    python generate_3d_map.py --nz 30 --h_atten 10.0 --h_bat 15.0
"""

import argparse
import os
import sys
import numpy as np

# ── ARGUMENTS ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Génération carte 3D synthétique")
parser.add_argument("--input",    default="damage_map.npz", help="Carte 2D (défaut: damage_map.npz)")
parser.add_argument("--output",   default="damage_map_3d.npz", help="Carte 3D (défaut: damage_map_3d.npz)")
parser.add_argument("--nz",       type=int,   default=30,   help="Nombre de niveaux en Z (défaut: 30)")
parser.add_argument("--z_max",    type=float, default=30.0, help="Hauteur max du domaine en m (défaut: 30.0)")
parser.add_argument("--h_atten",  type=float, default=10.0, help="Hauteur caractéristique d'atténuation en m (défaut: 10.0)")
parser.add_argument("--h_bat",    type=float, default=15.0, help="Hauteur des bâtiments en m (défaut: 15.0)")
args = parser.parse_args()

# ── CHARGEMENT CARTE 2D ──────────────────────────────────────────────────────
if not os.path.exists(args.input):
    print(f"ERREUR : {args.input} introuvable.")
    print("Lance d'abord :  python gp_damage.py export")
    sys.exit(1)

print(f"Chargement carte 2D : {args.input}")
gp = np.load(args.input)

xs            = gp['xs']              # (NX,)
ys            = gp['ys']              # (NY,)
damage_2d     = gp['damage_map']      # (NY, NX)
building_2d   = gp['building_mask']   # (NY, NX)  bool

NX, NY = len(xs), len(ys)
NZ     = args.nz
z_max  = args.z_max
H      = args.h_atten
h_bat  = args.h_bat

zs = np.linspace(0, z_max, NZ)
dz = zs[1] - zs[0] if NZ > 1 else z_max

print(f"  Grille 2D : {NX}×{NY} = {NX*NY:,} cellules")
print(f"  Extension 3D : NZ={NZ}, z_max={z_max:.1f}m, dz={dz:.2f}m")
print(f"  Atténuation : H={H:.1f}m  (score × exp(-z/H))")
print(f"  Hauteur bâtiments : {h_bat:.1f}m")

# ── CONSTRUCTION CARTE 3D ────────────────────────────────────────────────────
print("Construction de la carte 3D...")

# damage_3d[k, j, i] = danger au point (xs[i], ys[j], zs[k])
# Shape : (NZ, NY, NX)
damage_3d   = np.zeros((NZ, NY, NX), dtype=np.float64)
building_3d = np.zeros((NZ, NY, NX), dtype=bool)

for k in range(NZ):
    z = zs[k]

    # Atténuation exponentielle avec l'altitude
    attenuation = np.exp(-z / H)
    damage_3d[k, :, :] = damage_2d * attenuation

    # Bâtiments : bloqués jusqu'à h_bat
    if z <= h_bat:
        building_3d[k, :, :] = building_2d

print(f"  Carte 3D : shape={damage_3d.shape}")
print(f"  Cellules bâtiment 3D : {building_3d.sum():,} / {NX*NY*NZ:,}")

# Stats sur les scores
finite = damage_3d[~building_3d]
print(f"  Scores 3D : min={finite.min():.4f}  médiane={np.median(finite):.4f}"
      f"  max={finite.max():.4f}")

# Scores par niveau
print(f"\n  Profil vertical (score moyen hors bâtiments) :")
for k in range(0, NZ, max(1, NZ // 6)):
    layer = damage_3d[k, :, :]
    mask  = building_3d[k, :, :]
    free  = layer[~mask]
    if len(free) > 0:
        print(f"    z={zs[k]:5.1f}m  →  moy={free.mean():.4f}  max={free.max():.4f}")

# ── SAUVEGARDE ────────────────────────────────────────────────────────────────
np.savez(args.output,
         xs=xs,
         ys=ys,
         zs=zs,
         damage_map_3d=damage_3d,        # (NZ, NY, NX)
         building_mask_3d=building_3d,    # (NZ, NY, NX) bool
         # Garder les originaux 2D pour référence
         damage_map_2d=damage_2d,
         building_mask_2d=building_2d,
         # Paramètres
         h_atten=np.array([H]),
         h_bat=np.array([h_bat]),
)

size_mb = os.path.getsize(args.output) / 1e6 if os.path.exists(args.output) else 0
print(f"\nSauvegardé : {args.output}  ({size_mb:.1f} Mo)")
print(f"  Clés : xs, ys, zs, damage_map_3d, building_mask_3d,")
print(f"         damage_map_2d, building_mask_2d, h_atten, h_bat")
print(f"  Shape damage_map_3d : ({NZ}, {NY}, {NX})")