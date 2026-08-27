"""
gp_damage.py — GP interactif : carte de dégâts

Usage :
  python gp_damage.py add results/X31.3_Y38.0   # ajouter un résultat de simulation
  python gp_damage.py status                      # voir les points d'entraînement actuels
  python gp_damage.py map                         # régénérer la carte (sans ajouter de point)
  python gp_damage.py pmap 31.3 38.0             # carte de pression KB pour source (x,y)
  python gp_damage.py pmap results/X31.3_Y38.0  # KB + simulation réelle côte à côte
  python gp_damage.py reset                       # effacer tous les points d'entraînement

La carte est recalculée et sauvegardée dans damage_map.png après chaque commande add.
L'état (points d'entraînement) est persisté dans gp_state.npz.
"""

import sys, os, re
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

# ── Imports depuis kb.py et dammage.py ────────────────────────────────────
from kb import compute_pressure_for_source
from dammage import score_from_pressure

# =============================================================================
# Chemins
# =============================================================================
SCENE_DIR     = 'output-SCENE'
RESULTS_DIR   = 'results'
SONDES_PATH   = os.path.join(SCENE_DIR, 'sondes_batiments.vtk')
CONTOURS_PATH = os.path.join(SCENE_DIR, 'contours_frt.data')
STATE_FILE    = 'gp_state.npz'
W_TNT         = 40.0  # kg TNT

# =============================================================================
# Chargement de la scène (une seule fois au démarrage)
# =============================================================================
print("Chargement de la scène...", end=' ', flush=True)
_contours_raw = np.load(CONTOURS_PATH, allow_pickle=True)
buildings     = [np.asarray(_contours_raw[k], dtype=float) for k in _contours_raw.keys()]

sondes    = pv.read(SONDES_PATH)
sonde_pts = sondes.points  # (3127, 3)
_pd       = sondes.point_data
bat_ids   = np.array(_pd['bat_id'] if 'bat_id' in _pd else sondes.cell_data['bat_id'])
cats      = np.array(_pd['cat']    if 'cat'    in _pd else sondes.cell_data['cat'])
print("OK")

# =============================================================================
# Grille d'évaluation (100×100 sur [1,99]m)
# =============================================================================
NX, NY   = 100, 100
xs       = np.linspace(1.0, 99.0, NX)
ys       = np.linspace(1.0, 99.0, NY)
XX, YY   = np.meshgrid(xs, ys)
grid_pts = np.column_stack([XX.ravel(), YY.ravel()])  # (2500, 2)

# Masque bâtiments (cases infranchissables pour le chemin)
_building_mask = np.zeros((NY, NX), dtype=bool)
for poly in buildings:
    inside = mplpath.Path(poly).contains_points(grid_pts)
    _building_mask |= inside.reshape(NY, NX)

# =============================================================================
# Fonctions utilitaires
# =============================================================================

def kb_damage_score(source_xy):
    """Score de dégâts KB (surpression analytique) pour une source en (x,y)."""
    pmax_hpa = compute_pressure_for_source(source_xy, sonde_pts, buildings, W_TNT)
    return score_from_pressure(pmax_hpa, bat_ids, cats)


def true_damage_score(result_dir):
    """
    Score de dégâts réel extrait de results/X{x}_Y{y}/output-post.vtk.
    La pression stockée est absolue (Pa) avec base atmosphérique ~100000 Pa.
    On soustrait la baseline pour obtenir la surpression avant conversion en hPa.
    """
    from scipy.spatial import cKDTree

    mesh = pv.read(os.path.join(result_dir, 'last_map.vtk'))

    if 'pmax' in mesh.cell_data:
        pmax_pa = np.array(mesh.cell_data['pmax']).flatten()
        centers = mesh.cell_centers().points
    elif 'pmax' in mesh.point_data:
        pmax_pa = np.array(mesh.point_data['pmax']).flatten()
        centers = mesh.points
    else:
        keys = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
        raise ValueError(f"Champ 'pmax' introuvable dans {result_dir}. Clés : {keys}")

    pmax_hpa = (pmax_pa - 100_000.0) / 100.0  # Pa absolu → surpression hPa

    # Pour chaque sonde, trouver la cellule mesh la plus proche (en 2D)
    tree  = cKDTree(centers[:, :2])
    _, ix = tree.query(sonde_pts[:, :2])
    return score_from_pressure(pmax_hpa[ix], bat_ids, cats)


def build_kb_grid():
    """Calcule le score KB sur toute la grille (2500 points)."""
    print(f"Calcul KB sur {len(grid_pts)} points de grille...", end=' ', flush=True)
    scores = np.array([kb_damage_score(pt) for pt in grid_pts])
    print("OK")
    return scores


def fit_gp(train_xy, residuals):
    """Entraîne le GP sur les résidus (vrai - KB)."""
    kernel = (ConstantKernel(1.0, (1e-3, 10.0))
              * Matern(length_scale=20.0, length_scale_bounds=(5.0, 100.0), nu=2.5)
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1.0)))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(train_xy, residuals)
    return gp


def make_damage_map(train_xy=None, train_true=None, kb_grid=None):
    """
    Construit la carte de dégâts et retourne (damage_map, std_map).
    - Si aucun point d'entraînement : carte = KB uniquement.
    - Sinon : carte = KB + correction GP sur les résidus.
    """
    if kb_grid is None:
        kb_grid = build_kb_grid()

    if train_xy is None or len(train_xy) == 0:
        damage_map = kb_grid.reshape(NY, NX)
        std_map    = np.zeros((NY, NX))
        return damage_map, std_map, kb_grid

    train_kb   = np.array([kb_damage_score(pt) for pt in train_xy])
    residuals  = np.asarray(train_true) - train_kb
    gp         = fit_gp(np.asarray(train_xy), residuals)
    res_grid, std_grid = gp.predict(grid_pts, return_std=True)

    damage_map = (kb_grid + res_grid).reshape(NY, NX)
    std_map    = std_grid.reshape(NY, NX)
    return damage_map, std_map, kb_grid


# =============================================================================
# Sauvegarde / chargement de l'état GP
# =============================================================================

def load_state():
    if not os.path.exists(STATE_FILE):
        return np.empty((0, 2)), np.empty(0)
    d = np.load(STATE_FILE)
    return d['train_xy'], d['train_true']


def save_state(train_xy, train_true):
    np.savez(STATE_FILE, train_xy=train_xy, train_true=train_true)


# =============================================================================
# Visualisation commune
# =============================================================================

def plot_map(damage_map, std_map, train_xy, train_true, title, path_xy=None,
             point_a=None, point_b=None, filename='damage_map.png'):
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- carte des dégâts ---
    dm_plot = np.where(_building_mask, np.nan, damage_map)
    vmax    = np.nanmax(dm_plot) if not np.all(np.isnan(dm_plot)) else 1.0
    im0 = axes[0].pcolormesh(XX, YY, dm_plot, cmap='RdYlGn_r', vmin=0, vmax=vmax,
                             shading='auto')
    plt.colorbar(im0, ax=axes[0], label='Score de dégâts')

    for poly in buildings:
        p = np.array(poly)
        axes[0].fill(p[:, 0], p[:, 1], color='dimgray', alpha=0.6, zorder=3)
        axes[0].plot(list(p[:, 0]) + [p[0, 0]], list(p[:, 1]) + [p[0, 1]],
                     'k-', lw=1.2, zorder=4)

    if len(train_xy) > 0:
        axes[0].scatter(train_xy[:, 0], train_xy[:, 1], c=train_true,
                        cmap='RdYlGn_r', vmin=0, vmax=vmax,
                        s=100, edgecolors='k', lw=1.5, zorder=6,
                        label=f'Simulations ({len(train_xy)})')
        axes[0].legend(fontsize=8)

    if path_xy is not None:
        axes[0].plot(path_xy[:, 0], path_xy[:, 1], 'b-', lw=2.5,
                     zorder=7, label='Chemin optimal')
    if point_a:
        axes[0].scatter(*point_a, c='lime',  s=200, marker='o',
                        edgecolors='k', lw=1.5, zorder=8, label=f'A {point_a}')
    if point_b:
        axes[0].scatter(*point_b, c='gold',  s=200, marker='s',
                        edgecolors='k', lw=1.5, zorder=8, label=f'B {point_b}')
    axes[0].set_title(title)
    axes[0].set_xlabel('x (m)'); axes[0].set_ylabel('y (m)')
    axes[0].set_xlim(0, 100);   axes[0].set_ylim(0, 100)
    if path_xy is not None or point_a:
        axes[0].legend(fontsize=8)

    # --- carte d'incertitude ---
    im1 = axes[1].pcolormesh(XX, YY, np.where(_building_mask, np.nan, std_map),
                             cmap='Blues', shading='auto')
    plt.colorbar(im1, ax=axes[1], label='Incertitude GP (σ)')
    for poly in buildings:
        p = np.array(poly)
        axes[1].fill(p[:, 0], p[:, 1], color='dimgray', alpha=0.6, zorder=3)
        axes[1].plot(list(p[:, 0]) + [p[0, 0]], list(p[:, 1]) + [p[0, 1]],
                     'k-', lw=1.2, zorder=4)
    if len(train_xy) > 0:
        axes[1].scatter(train_xy[:, 0], train_xy[:, 1], c='red', s=80, zorder=6)
    axes[1].set_title('Incertitude GP (σ)')
    axes[1].set_xlabel('x (m)'); axes[1].set_ylabel('y (m)')
    axes[1].set_xlim(0, 100);   axes[1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Carte sauvegardée : {filename}")


# =============================================================================
# Commandes CLI
# =============================================================================

def cmd_add(result_dir):
    """Ajoute un résultat de simulation et met à jour la carte."""
    m = re.search(r'X([\d.]+)_Y([\d.]+)', result_dir)
    if not m:
        print(f"Impossible d'extraire (x,y) depuis '{result_dir}'")
        sys.exit(1)
    x, y = float(m.group(1)), float(m.group(2))

    print(f"Ajout de la simulation X={x} Y={y}...")
    true_sc = true_damage_score(result_dir)
    kb_sc   = kb_damage_score((x, y))
    print(f"  Score réel   : {true_sc:.4f}")
    print(f"  Score KB     : {kb_sc:.4f}")
    print(f"  Résidu       : {true_sc - kb_sc:+.4f}")

    train_xy, train_true = load_state()
    train_xy   = np.vstack([train_xy,   [[x, y]]]) if len(train_xy)   else np.array([[x, y]])
    train_true = np.append(train_true, true_sc)
    save_state(train_xy, train_true)
    print(f"État sauvegardé ({len(train_true)} point(s) au total)")

    damage_map, std_map, _ = make_damage_map(train_xy, train_true)
    title = (f'Carte GP — {len(train_true)} simulation(s)\n'
             f'Dernier ajout : X={x} Y={y}  score={true_sc:.3f}')
    plot_map(damage_map, std_map, train_xy, train_true, title)


def cmd_status():
    """Affiche les points d'entraînement actuels."""
    train_xy, train_true = load_state()
    if len(train_xy) == 0:
        print("Aucun point d'entraînement. Utilisez : python gp_damage.py add results/X{x}_Y{y}")
        return
    print(f"\n{'#':>3}  {'x':>7}  {'y':>7}  {'score_vrai':>12}  {'score_KB':>10}  {'résidu':>8}")
    print("-" * 55)
    for i, ((x, y), sc) in enumerate(zip(train_xy, train_true)):
        kb_sc = kb_damage_score((x, y))
        print(f"{i+1:>3}  {x:>7.2f}  {y:>7.2f}  {sc:>12.4f}  {kb_sc:>10.4f}  {sc-kb_sc:>+8.4f}")
    print(f"\n{len(train_xy)} point(s) au total.")


def cmd_map():
    """Régénère la carte avec les points d'entraînement actuels."""
    train_xy, train_true = load_state()
    n = len(train_xy)
    print(f"Régénération de la carte avec {n} point(s)...")
    damage_map, std_map, _ = make_damage_map(
        train_xy if n else None,
        train_true if n else None
    )
    title = f'Carte GP — {n} simulation(s)' if n else 'Carte KB (aucune simulation)'
    plot_map(damage_map, std_map, train_xy, train_true, title)


def cmd_pmap(arg):
    """
    Carte de pression en hPa pour une source donnée.
    arg peut être :
      - "31.3 38.0"            → KB uniquement
      - "results/X31.3_Y38.0" → KB + simulation réelle côte à côte
    """
    from scipy.spatial import cKDTree

    result_dir = None
    m = re.search(r'X([\d.]+)_Y([\d.]+)', arg)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if os.path.isdir(arg):
            result_dir = arg
    else:
        try:
            parts = arg.split()
            x, y = float(parts[0]), float(parts[1])
        except Exception:
            print(f"Argument invalide : '{arg}'")
            print("Usage : python gp_damage.py pmap 31.3 38.0")
            print("     ou python gp_damage.py pmap results/X31.3_Y38.0")
            sys.exit(1)

    source_xy = np.array([x, y])

    # --- Pression KB sur la grille ---
    print(f"Calcul KB sur {len(grid_pts)} points pour source ({x}, {y})...", end=' ', flush=True)
    pmax_kb = compute_pressure_for_source(source_xy, grid_pts, buildings, W_TNT)
    pmax_kb_grid = pmax_kb.reshape(NY, NX)
    print("OK")

    # --- Pression réelle depuis last_map.vtk (si dossier fourni) ---
    pmax_real_grid = None
    if result_dir is not None:
        vtk_path = os.path.join(result_dir, 'last_map.vtk')
        print(f"Chargement {vtk_path}...", end=' ', flush=True)
        mesh = pv.read(vtk_path)
        if 'pmax' in mesh.cell_data:
            pmax_pa = np.array(mesh.cell_data['pmax']).flatten()
            centers = mesh.cell_centers().points
        else:
            pmax_pa = np.array(mesh.point_data['pmax']).flatten()
            centers = mesh.points
        pmax_real_hpa = (pmax_pa - 100_000.0) / 100.0

        # Interpolation sur la grille régulière via plus proche voisin
        tree = cKDTree(centers[:, :2])
        _, ix = tree.query(grid_pts)
        pmax_real_grid = pmax_real_hpa[ix].reshape(NY, NX)
        print("OK")

    # --- Tracé ---
    has_real = pmax_real_grid is not None
    ncols = 3 if has_real else 1
    _, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    mask = _building_mask

    def _draw_buildings(ax):
        for poly in buildings:
            p = np.array(poly)
            ax.fill(p[:, 0], p[:, 1], color='dimgray', alpha=0.7, zorder=3)
            ax.plot(list(p[:, 0]) + [p[0, 0]], list(p[:, 1]) + [p[0, 1]],
                    'k-', lw=1.2, zorder=4)
        ax.scatter(x, y, c='cyan', s=200, marker='*', zorder=5, label='Source')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_xlim(0, 100);   ax.set_ylim(0, 100)
        ax.legend(fontsize=8)

    # Panneau 1 — KB (échelle log)
    d_kb = np.where(mask, np.nan, pmax_kb_grid)
    vmin_log = max(np.nanpercentile(d_kb, 2), 0.1)
    vmax_log = max(np.nanpercentile(d_kb, 98), 1.0)
    norm_log = matplotlib.colors.LogNorm(vmin=vmin_log, vmax=vmax_log)
    im = axes[0].pcolormesh(XX, YY, d_kb, cmap='hot_r', norm=norm_log, shading='auto')
    plt.colorbar(im, ax=axes[0], label='Pression (hPa, log)')
    axes[0].set_title(f'KB (analytique, log)\nSource ({x}, {y})')
    _draw_buildings(axes[0])

    if has_real:
        # Panneau 2 — Simulation réelle (échelle log)
        d_real = np.where(mask, np.nan, pmax_real_grid)
        vmin_log2 = max(np.nanpercentile(d_real[~np.isnan(d_real)], 2), 0.1)
        vmax_log2 = max(np.nanpercentile(d_real[~np.isnan(d_real)], 98), 1.0)
        norm_log2 = matplotlib.colors.LogNorm(vmin=vmin_log2, vmax=vmax_log2)
        im = axes[1].pcolormesh(XX, YY, d_real, cmap='hot_r', norm=norm_log2, shading='auto')
        plt.colorbar(im, ax=axes[1], label='Pression (hPa, log)')
        axes[1].set_title(f'Simulation réelle (last_map.vtk, log)\nSource ({x}, {y})')
        _draw_buildings(axes[1])

        # Panneau 3 — Résidu (réel − KB), échelle linéaire centrée sur 0
        residual = np.where(mask, np.nan, pmax_real_grid - pmax_kb_grid)
        absmax = max(np.nanpercentile(np.abs(residual[~np.isnan(residual)]), 98), 1.0)
        im = axes[2].pcolormesh(XX, YY, residual, cmap='RdBu_r', shading='auto',
                                vmin=-absmax, vmax=absmax)
        plt.colorbar(im, ax=axes[2], label='Résidu (hPa)')
        axes[2].set_title(f'Résidu réel − KB\nSource ({x}, {y})')
        _draw_buildings(axes[2])

    plt.tight_layout()
    fname = f'pmap_X{x}_Y{y}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Carte sauvegardée : {fname}")


def cmd_export():
    """
    Exporte la carte de dégâts GP sous deux formats :
      - damage_map.npz  : arrays numpy (xs, ys, damage_map, std_map, building_mask)
      - damage_map.vtk  : grille structurée lisible par ParaView / pyvista
    """
    train_xy, train_true = load_state()
    n = len(train_xy)
    print(f"Calcul de la carte GP avec {n} point(s)...")
    damage_map, std_map, _ = make_damage_map(
        train_xy if n else None,
        train_true if n else None
    )

    # --- npz ---
    np.savez('damage_map.npz',
             xs=xs, ys=ys,
             damage_map=damage_map,
             std_map=std_map,
             building_mask=_building_mask)
    print("Exporté : damage_map.npz")
    print("  Clés : xs (100,)  ys (100,)  damage_map (100,100)  std_map (100,100)  building_mask (100,100)")

    # --- vtk : StructuredGrid ---
    z_zero = np.zeros_like(XX)
    grid = pv.StructuredGrid(XX, YY, z_zero)
    flat = damage_map.ravel(order='C')
    std_flat = std_map.ravel(order='C')
    mask_flat = _building_mask.ravel(order='C').astype(float)
    grid.point_data['damage_score'] = flat
    grid.point_data['gp_std']       = std_flat
    grid.point_data['building']     = mask_flat
    grid.save('damage_map.vtk')
    print("Exporté : damage_map.vtk")
    print("  Champs : damage_score, gp_std, building")


def cmd_loocv():
    """
    Leave-One-Out Cross-Validation sur tous les points d'entraînement.
    Pour chaque point i : entraîne le GP sur les N-1 autres, prédit en i.
    Génère un PNG par fold + un PNG de synthèse.
    """
    train_xy, train_true = load_state()
    N = len(train_xy)
    if N < 2:
        print("Il faut au moins 2 points pour faire une LOOCV.")
        return

    print(f"\nLOOCV sur {N} points...")
    kb_all    = np.array([kb_damage_score(pt) for pt in train_xy])
    residuals_all = train_true - kb_all

    preds   = np.zeros(N)
    stds    = np.zeros(N)
    kb_grid = build_kb_grid()

    for i in range(N):
        mask_loo = np.ones(N, dtype=bool)
        mask_loo[i] = False

        xy_loo   = train_xy[mask_loo]
        res_loo  = residuals_all[mask_loo]

        gp = fit_gp(xy_loo, res_loo)
        pred_res, std = gp.predict(train_xy[i:i+1], return_std=True)
        preds[i] = kb_all[i] + pred_res[0]
        stds[i]  = std[0]

        # Carte du fold i
        res_grid, std_grid = gp.predict(grid_pts, return_std=True)
        dm  = (kb_grid + res_grid).reshape(NY, NX)
        sm  = std_grid.reshape(NY, NX)

        _, axes = plt.subplots(1, 2, figsize=(14, 6))
        xi, yi = train_xy[i]

        # panneau gauche : carte dégâts
        dm_plot = np.where(_building_mask, np.nan, dm)
        vmax = max(np.nanmax(dm_plot[~np.isnan(dm_plot)]), 1.0)
        axes[0].pcolormesh(XX, YY, dm_plot, cmap='RdYlGn_r',
                           vmin=0, vmax=vmax, shading='auto')
        # points d'entraînement utilisés
        axes[0].scatter(xy_loo[:, 0], xy_loo[:, 1],
                        c='white', s=80, edgecolors='k', lw=1.2,
                        zorder=6, label='Train')
        # point laissé de côté
        err = preds[i] - train_true[i]
        col = 'lime' if abs(err) < 1.0 else ('orange' if abs(err) < 3.0 else 'red')
        axes[0].scatter(xi, yi, c=col, s=200, marker='*',
                        edgecolors='k', lw=1.5, zorder=7,
                        label=f'LOO  vrai={train_true[i]:.2f}  prédit={preds[i]:.2f}  err={err:+.2f}')
        for poly in buildings:
            p = np.array(poly)
            axes[0].fill(p[:, 0], p[:, 1], color='dimgray', alpha=0.6, zorder=3)
            axes[0].plot(list(p[:, 0]) + [p[0, 0]], list(p[:, 1]) + [p[0, 1]],
                         'k-', lw=1.2, zorder=4)
        axes[0].set_title(f'Fold {i+1}/{N} — LOO : X={xi} Y={yi}\n'
                          f'vrai={train_true[i]:.3f}  prédit={preds[i]:.3f}  σ={stds[i]:.3f}')
        axes[0].set_xlabel('x (m)'); axes[0].set_ylabel('y (m)')
        axes[0].set_xlim(0, 100);   axes[0].set_ylim(0, 100)
        axes[0].legend(fontsize=8)

        # panneau droit : incertitude
        axes[1].pcolormesh(XX, YY, np.where(_building_mask, np.nan, sm),
                           cmap='Blues', shading='auto')
        axes[1].scatter(xy_loo[:, 0], xy_loo[:, 1],
                        c='red', s=80, zorder=6)
        axes[1].scatter(xi, yi, c=col, s=200, marker='*',
                        edgecolors='k', lw=1.5, zorder=7)
        for poly in buildings:
            p = np.array(poly)
            axes[1].fill(p[:, 0], p[:, 1], color='dimgray', alpha=0.6, zorder=3)
            axes[1].plot(list(p[:, 0]) + [p[0, 0]], list(p[:, 1]) + [p[0, 1]],
                         'k-', lw=1.2, zorder=4)
        axes[1].set_title(f'Incertitude GP — fold {i+1}/{N}')
        axes[1].set_xlabel('x (m)'); axes[1].set_ylabel('y (m)')
        axes[1].set_xlim(0, 100);   axes[1].set_ylim(0, 100)

        plt.tight_layout()
        fname = f'loocv_fold{i+1:02d}_X{xi}_Y{yi}.png'
        plt.savefig(fname, dpi=150)
        plt.close()
        err_str = f'err={err:+.3f}'
        print(f"  Fold {i+1:2d} — vrai={train_true[i]:.3f}  prédit={preds[i]:.3f}  σ={stds[i]:.3f}  {err_str}  → {fname}")

    # --- PNG de synthèse ---
    errors = preds - train_true
    mae    = np.mean(np.abs(errors))
    rmse   = np.sqrt(np.mean(errors**2))

    _, axes = plt.subplots(1, 2, figsize=(13, 5))

    # scatter vrai vs prédit
    vmin_s = min(train_true.min(), preds.min()) - 0.5
    vmax_s = max(train_true.max(), preds.max()) + 0.5
    axes[0].plot([vmin_s, vmax_s], [vmin_s, vmax_s], 'k--', lw=1, label='Parfait')
    sc = axes[0].scatter(train_true, preds, c=np.abs(errors),
                         cmap='RdYlGn_r', vmin=0, vmax=max(np.abs(errors).max(), 1),
                         s=100, edgecolors='k', lw=1, zorder=3)
    plt.colorbar(sc, ax=axes[0], label='|erreur|')
    for i, (xt, xp) in enumerate(zip(train_true, preds)):
        axes[0].annotate(f'  ({train_xy[i,0]:.0f},{train_xy[i,1]:.0f})',
                         (xt, xp), fontsize=7)
    axes[0].set_xlabel('Score réel'); axes[0].set_ylabel('Score prédit (LOO)')
    axes[0].set_title(f'LOOCV — MAE={mae:.3f}  RMSE={rmse:.3f}')
    axes[0].legend(fontsize=8)

    # barplot erreurs
    colors = ['green' if abs(e) < 1 else ('orange' if abs(e) < 3 else 'red')
              for e in errors]
    labels = [f'({train_xy[i,0]:.0f},{train_xy[i,1]:.0f})' for i in range(N)]
    axes[1].bar(labels, errors, color=colors)
    axes[1].axhline(0, color='k', lw=1)
    axes[1].axhline( mae, color='gray', lw=1, ls='--', label=f'±MAE={mae:.2f}')
    axes[1].axhline(-mae, color='gray', lw=1, ls='--')
    axes[1].set_xlabel('Source'); axes[1].set_ylabel('Erreur (prédit − réel)')
    axes[1].set_title('Erreurs LOO par point')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('loocv_summary.png', dpi=150)
    plt.close()
    print(f"\n  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"Synthèse sauvegardée : loocv_summary.png")


def cmd_reset():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print("État réinitialisé.")
    else:
        print("Aucun état à réinitialiser.")


# =============================================================================
# Point d'entrée
# =============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == 'add':
        if len(sys.argv) < 3:
            print("Usage : python gp_damage.py add results/X{x}_Y{y}")
            sys.exit(1)
        cmd_add(sys.argv[2])

    elif cmd == 'status':
        cmd_status()

    elif cmd == 'map':
        cmd_map()

    elif cmd == 'pmap':
        if len(sys.argv) < 3:
            print("Usage : python gp_damage.py pmap <x> <y>")
            print("     ou python gp_damage.py pmap results/X{x}_Y{y}")
            sys.exit(1)
        # accepte soit "pmap 31.3 38.0" soit "pmap results/X31.3_Y38.0"
        arg = ' '.join(sys.argv[2:]) if not sys.argv[2].startswith('results') else sys.argv[2]
        cmd_pmap(arg)

    elif cmd == 'export':
        cmd_export()

    elif cmd == 'loocv':
        cmd_loocv()

    elif cmd == 'reset':
        cmd_reset()

    else:
        print(f"Commande inconnue : '{cmd}'")
        print(__doc__)
        sys.exit(1)