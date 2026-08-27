import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ── Kingery-Bulmash polynômes (Swisdak 1994, surface burst, métrique) ──────
# pression incidente en kPa, Z en m/kg^(1/3)
# Y = somme Ci * (log10(Z))^i
KB_COEFS = [
    2.611368,
   -1.699705,
    0.154576,
    0.014999,
   -0.138976,
    0.004684,
    0.026525,
   -0.011899,
   -0.001445,
]
KB_Z_MIN = 0.2   # m/kg^(1/3)
KB_Z_MAX = 40.0  # m/kg^(1/3)


def kb_incident_pressure_kpa(Z):
    Z = np.asarray(Z, dtype=float)
    Z_clipped = np.clip(Z, KB_Z_MIN, KB_Z_MAX)
    logZ = np.log10(Z_clipped)
    Y = sum(c * logZ**i for i, c in enumerate(KB_COEFS))
    return 10**Y

def kb_pmax_hpa(R_m, W_kg):
    """
    Pression maximale (hPa) à distance R (m) pour charge W kg TNT.
    """
    Z = R_m / W_kg**(1/3)
    return kb_incident_pressure_kpa(Z) * 10  # kPa -> hPa


# ── Raycasting 2D ──────────────────────────────────────────────────────────
def segments_intersect(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0]*d2[1] - d1[1]*d2[0]
    if abs(cross) < 1e-10:
        return False
    t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / cross
    u = ((p3[0]-p1[0])*d1[1] - (p3[1]-p1[1])*d1[0]) / cross
    return 0 < t < 1 and 0 < u < 1


def count_walls(point, source, buildings):
    p1 = source[:2]
    p2 = point[:2]
    count = 0
    for poly in buildings:
        n = len(poly)
        for i in range(n):
            p3 = np.array(poly[i])
            p4 = np.array(poly[(i+1) % n])
            if segments_intersect(p1, p2, p3, p4):
                count += 1
    return count


# ── Paramètres ─────────────────────────────────────────────────────────────
SOURCE        = np.array([30.0, 50.0])
CONTOURS_PATH = '../../01-SCENE/output-SCENE/contours_frt.data'
SCENE_PATH    = '../../03-POST/2D/output-post.vtk'
P0 = 1200
def shadow_factor(dist):
    """Plus proche = plus atténué derrière un mur."""
    return np.clip(0.3 + 0.25 * (dist / 70), 0.3, 0.8)

# ── Chargement ─────────────────────────────────────────────────────────────
contours_raw = np.load(CONTOURS_PATH, allow_pickle=True)
buildings    = [np.array(contours_raw[k]) for k in contours_raw.keys()]

mesh      = pv.read(SCENE_PATH)
centers   = mesh.cell_centers().points
pmax_real = mesh.cell_data['pmax'] / 100  # Pa -> hPa

dist = np.sqrt((centers[:,0]-SOURCE[0])**2 + (centers[:,1]-SOURCE[1])**2)
dist = np.clip(dist, 0.5, None)

# ── Fit W (masse TNT équivalente) sur les données réelles ──────────────────
print('Fit de W (masse TNT équivalente)...')

def rmse_log(logW):
    W = 10**logW
    pmax_kb = kb_pmax_hpa(dist, W)+P0
    # RMSE en log pour ne pas être dominé par les valeurs extrêmes
    return np.sqrt(np.mean((np.log10(pmax_kb + 1) - np.log10(pmax_real + 1))**2))

result = minimize_scalar(rmse_log, bounds=(-2, 8), method='bounded')
W_fit = 10**result.x
print(f'W TNT équivalent = {W_fit:.2f} kg')

# ── Simulation avec KB + raycasting ───────────────────────────────────────
print('Raycasting...')
pmax_estim = kb_pmax_hpa(dist, 10e3).copy()

for i, pt in enumerate(centers):
    if i % 500 == 0:
        print(f'  {i}/{len(centers)}')
    n_walls = count_walls(pt, SOURCE, buildings)
    if n_walls > 0:
        pmax_estim[i] *= shadow_factor(dist[i]) ** n_walls

# ── Sauvegarde ─────────────────────────────────────────────────────────────
mesh.cell_data['pmax_estim_hpa'] = pmax_estim
mesh.save('output_simplifie.vtk')

# ── Visualisation ──────────────────────────────────────────────────────────
vmin, vmax = 1e2, 1e7

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc1 = axes[0].scatter(centers[:,0], centers[:,1], c=pmax_real,
                      cmap='coolwarm', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), s=2)
axes[0].scatter(*SOURCE, c='blue', s=200, marker='*', zorder=5)
plt.colorbar(sc1, ax=axes[0], label='pmax réelle (hPa)')
axes[0].set_title('Simulation ARMEN')

sc2 = axes[1].scatter(centers[:,0], centers[:,1], c=pmax_estim,
                      cmap='coolwarm', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), s=2)
axes[1].scatter(*SOURCE, c='blue', s=200, marker='*', zorder=5)
plt.colorbar(sc2, ax=axes[1], label='pmax estimée KB (hPa)')
axes[1].set_title(f'Kingery-Bulmash (W={W_fit:.1f} kg TNT) + raycasting')

plt.tight_layout()
plt.savefig('comparaison.png', dpi=150)
print('comparaison.png généré')

# ── Erreur ────────────────────────────────────────────────────────────────
erreur = (pmax_estim - pmax_real) / pmax_real
fig2, ax = plt.subplots(figsize=(10, 8))
sc3 = ax.scatter(centers[:,0], centers[:,1], c=erreur,
                 cmap='RdBu_r', vmin=-2, vmax=2, s=2)
ax.scatter(*SOURCE, c='black', s=200, marker='*', zorder=5)
plt.colorbar(sc3, ax=ax, label='erreur relative (estim-réel)/réel')
ax.set_title('Erreur relative')
plt.tight_layout()
plt.savefig('erreur.png', dpi=150)

erreur_abs = np.abs(erreur)
print(f'Erreur médiane : {np.median(erreur_abs)*100:.1f}%')
print(f'Erreur moyenne : {np.mean(erreur_abs)*100:.1f}%')
print(f'Erreur p75     : {np.percentile(erreur_abs, 75)*100:.1f}%')
print(f'Erreur p90     : {np.percentile(erreur_abs, 90)*100:.1f}%')
print('OK')