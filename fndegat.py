"""
Score de dégâts MATÉRIELS des bâtiments
Fichier : output-post.vtk  |  Champ : pmax (en Pa → converti en hPa)
Seuils ICPE - effets sur les structures
"""

import numpy as np
import pyvista as pv

# =============================================================================
# CONFIG
# =============================================================================

VTK_FILE    = "../../03-POST/2D/output-post.vtk"  
OUTPUT_FILE = "degats_materiels.vtk"

FIELD_NAME  = "pmax"              # nom du champ dans le VTK
PA_TO_HPA   = 1 / 100             # conversion Pa → hPa

# =============================================================================
# SEUILS ICPE STRUCTURES (hPa)
#
#  Niveau │ Seuil  │ Description
# ────────┼────────┼──────────────────────────────────────────
#    0    │  < 20  │ Aucun dégât
#    1    │ ≥  20  │ Destructions significatives de vitres
#    2    │ ≥  50  │ Dégâts légers sur structures
#    3    │ ≥ 140  │ Dégâts graves sur structures
#    4    │ ≥ 200  │ Effets domino
#    5    │ ≥ 300  │ Dégâts très graves sur structures
# =============================================================================

NIVEAUX = [
    (300, 5, "Dégâts très graves"),
    (200, 4, "Effets domino"),
    (140, 3, "Dégâts graves"),
    ( 50, 2, "Dégâts légers"),
    ( 20, 1, "Bris de vitres"),
    (  0, 0, "Aucun dégât"),
]

# =============================================================================
# CHARGEMENT
# =============================================================================

print("\n" + "="*60)
print("  SCORE DÉGÂTS MATÉRIELS — output-post.vtk (pmax)")
print("="*60)

mesh = pv.read(VTK_FILE)
print(f"\nType     : {type(mesh).__name__}")
print(f"Points   : {mesh.n_points:,}")
print(f"Cellules : {mesh.n_cells:,}")

# Récupération du champ pmax
if FIELD_NAME in mesh.point_data:
    pmax_pa = np.array(mesh.point_data[FIELD_NAME]).flatten()
    loc = "point"
elif FIELD_NAME in mesh.cell_data:
    pmax_pa = np.array(mesh.cell_data[FIELD_NAME]).flatten()
    loc = "cell"
else:
    available = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
    raise ValueError(f"Champ '{FIELD_NAME}' introuvable. Disponibles : {available}")

# Conversion Pa → hPa
pmax_hpa = pmax_pa * PA_TO_HPA

print(f"\nChamp '{FIELD_NAME}' ({loc} data)")
print(f"  Pression min : {pmax_pa.min():.0f} Pa  ({pmax_hpa.min():.1f} hPa)")
print(f"  Pression max : {pmax_pa.max():.0f} Pa  ({pmax_hpa.max():.1f} hPa)")
print(f"  Pression moy : {pmax_pa.mean():.0f} Pa  ({pmax_hpa.mean():.1f} hPa)")

# =============================================================================
# CALCUL DU SCORE
# =============================================================================

score = np.zeros(len(pmax_hpa), dtype=np.int32)
for seuil, niveau, _ in reversed(NIVEAUX):
    score[pmax_hpa >= seuil] = niveau
# =============================================================================
# RÉSULTATS
# =============================================================================

n = len(pmax_hpa)
print(f"\n{'Niv':>4}  {'Seuil':>8}  {'Désignation':<35}  {'Points':>9}  {'%':>6}")
print("-" * 70)

for seuil, niveau, desc in reversed(NIVEAUX):
    n_pts = int(np.sum(score == niveau))
    pct   = 100 * n_pts / n
    seuil_str = f">={seuil} hPa" if seuil > 0 else "< 20 hPa"
    print(f"{niveau:>4}  {seuil_str:>8}  {desc:<35}  {n_pts:>9,}  {pct:>5.1f}%")

print("-" * 70)
print(f"{'TOTAL':>4}  {'':>8}  {'':35}  {n:>9,}  100.0%\n")

# Zones critiques
for seuil, niveau, desc in NIVEAUX[:-2]:
    n_pts = int(np.sum(score >= niveau))
    pct   = 100 * n_pts / n
    print(f"[!] >= niveau {niveau} ({desc}) : {n_pts:,} pts ({pct:.1f}%)")

# =============================================================================
# EXPORT VTK ENRICHI
# =============================================================================

if loc == "point":
    mesh.point_data["score_degats"] = score
    mesh.point_data["pmax_hpa"]     = pmax_hpa
else:
    mesh.cell_data["score_degats"]  = score
    mesh.cell_data["pmax_hpa"]      = pmax_hpa

mesh.save(OUTPUT_FILE)
print(f"\n[OK] Fichier exporte : {OUTPUT_FILE}")
print("  Nouveaux champs : 'score_degats' (0-5) et 'pmax_hpa'")
print("  -> ParaView : colorier par 'score_degats', colormap RdYlGn inversee, clim [0,5]")

# =============================================================================
# VISUALISATION (decommenter si affichage disponible sur le cluster)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
def visu_degats(mesh, output_file="degats_materiels.png", source=(30.0, 50.0)):
    score = mesh.cell_data["score_degats"]
    points = mesh.cell_centers().points

    couleurs = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red', 4: 'darkred', 5: 'black'}

    plt.figure(figsize=(10, 8))
    for niveau, couleur in couleurs.items():
        mask = score == niveau
        if mask.sum() > 0:
            plt.scatter(points[mask, 0], points[mask, 1], c=couleur, s=2)

    patches = [
        mpatches.Patch(color='green',   label='0 - Aucun dégât (< 20 hPa)'),
        mpatches.Patch(color='yellow',  label='1 - Bris de vitres (≥ 20 hPa)'),
        mpatches.Patch(color='orange',  label='2 - Dégâts légers (≥ 50 hPa)'),
        mpatches.Patch(color='red',     label='3 - Dégâts graves (≥ 140 hPa)'),
        mpatches.Patch(color='darkred', label='4 - Effets domino (≥ 200 hPa)'),
        mpatches.Patch(color='black',   label='5 - Très graves (≥ 300 hPa)'),
    ]
    plt.legend(handles=patches, loc='lower right', fontsize=7)
    plt.scatter(*source, c='blue', s=200, marker='*', zorder=5, label='Source')
    plt.title('Score de dégâts matériels - Seuils ICPE')
    plt.savefig(output_file, dpi=150)
    print(f"[OK] Image exportée : {output_file}")
visu_degats(mesh)

sondes_path = '../../01-SCENE/output-SCENE/sondes_batiments.vtk'

# Poids par catégorie
POIDS_CAT = {1: 1.0, 2: 2.0, 3: 3.0, 4: 5.0}
P_MIN = 20    # seuil bas hPa
P_MAX = 300   # seuil destruction totale hPa

def score_cellule(pmax_hpa, poids):
    p = np.asarray(pmax_hpa, dtype=float)
    s = np.zeros_like(p)
    mask = p >= P_MIN
    s[mask] = np.log(np.clip(p[mask], P_MIN, P_MAX) / P_MIN) / np.log(P_MAX / P_MIN)
    s = np.clip(s, 0, 1) * poids
    return s

def score_explosion(mesh_post, sondes_path):
    from scipy.spatial import cKDTree
    sondes = pv.read(sondes_path)
    bat_ids = sondes.cell_data['bat_id']
    cats    = sondes.cell_data['cat']

    centers  = mesh_post.cell_centers().points
    pmax_hpa = mesh_post.cell_data['pmax_hpa']

    tree = cKDTree(sondes.points)
    _, idx = tree.query(centers)

    bat_id_par_cellule = bat_ids[idx]
    cat_par_cellule    = cats[idx]
    poids_par_cellule  = np.array([POIDS_CAT.get(int(c), 1.0) for c in cat_par_cellule])
    scores_cellules    = score_cellule(pmax_hpa, poids_par_cellule)

    scores_bat  = {}
    cat_par_bat = {}
    for bid in np.unique(bat_id_par_cellule):
        mask = bat_id_par_cellule == bid
        scores_bat[int(bid)]  = float(np.mean(scores_cellules[mask]))
        cat_par_bat[int(bid)] = int(cat_par_cellule[mask][0])

    score_global = sum(scores_bat.values())

    print("\n=== SCORE DE DÉGÂTS DE L'EXPLOSION ===")
    print(f"{'Bât':>4}  {'Cat':>4}  {'Poids':>6}  {'Score':>8}")
    print("-" * 35)
    for bid, sc in sorted(scores_bat.items()):
        cat   = cat_par_bat[bid]
        poids = POIDS_CAT.get(cat, 1.0)
        print(f"{bid:>4}  {cat:>4}  {poids:>6.1f}  {sc:>8.4f}")
    print("-" * 35)
    print(f"{'TOTAL':>16}  {score_global:>8.4f}")

    return score_global, scores_bat, cat_par_bat

score_global, scores_bat, cat_par_bat = score_explosion(mesh, sondes_path)