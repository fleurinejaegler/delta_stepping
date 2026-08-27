"""
visualize_chemin_3d.py
Exporte la topographie + chemin optimal en fichiers VTK pour ParaView.

Sortie :
  - terrain_surface.vtk   : surface du terrain (points 3D avec Z = élévation)
  - batiments_3d.vtk      : volumes des bâtiments
  - chemin_optimal_3d.vtk  : polyligne du chemin

Option --preview : ouvre une fenêtre pyvista interactive

Usage :
    python3 visualize_chemin_3d.py
    python3 visualize_chemin_3d.py --preview
"""

import argparse
import numpy as np

# ── ARGUMENTS ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Export terrain + chemin → VTK")
parser.add_argument("--result",         default="result_3d.npz")
parser.add_argument("--meta",           default="meta_3d.npz")
parser.add_argument("--damage",         default="damage_map_3d.npz")
parser.add_argument("--out_terrain",    default="terrain_surface.vtk")
parser.add_argument("--out_buildings",  default="batiments_3d.vtk")
parser.add_argument("--out_chemin",     default="chemin_optimal_3d.vtk")
parser.add_argument("--preview",        action="store_true",
                    help="Ouvrir une fenêtre pyvista interactive")
args = parser.parse_args()

# ── CHARGEMENT ────────────────────────────────────────────────────────────────
print("Chargement des données...")
res  = np.load(args.result, allow_pickle=True)
meta = np.load(args.meta, allow_pickle=True)
dmg  = np.load(args.damage, allow_pickle=True)

path       = res['path']
dist       = res['dist']
source     = int(res['source'][0])
target     = int(res['target'][0])

centers    = meta['centers']
node_score = meta['node_score']
xs, ys, zs = meta['xs'], meta['ys'], meta['zs']
NX = int(meta['NX'][0])
NY = int(meta['NY'][0])
NZ = int(meta['NZ'][0])

terrain      = dmg['terrain_elevation']    # (NY, NX)
building_3d  = dmg['building_mask_3d']     # (NZ, NY, NX)
building_2d  = dmg['building_mask_2d']     # (NY, NX)
h_bat        = float(dmg['h_bat'][0])

print(f"Grille : {NX}×{NY}×{NZ}")
print(f"Terrain : [{terrain.min():.2f}, {terrain.max():.2f}]m")
print(f"Chemin : {len(path)} nœuds, danger cumulé = {dist[target]:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT 1 : SURFACE TERRAIN (StructuredGrid ASCII)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nExport terrain : {args.out_terrain}")

with open(args.out_terrain, 'w') as f:
    f.write("# vtk DataFile Version 3.0\n")
    f.write("Terrain surface\n")
    f.write("ASCII\n")
    f.write("DATASET STRUCTURED_GRID\n")
    f.write(f"DIMENSIONS {NX} {NY} 1\n")
    f.write(f"POINTS {NX * NY} float\n")

    # Points 3D : (xs[i], ys[j], terrain[j,i])
    for j in range(NY):
        for i in range(NX):
            f.write(f"{xs[i]:.6f} {ys[j]:.6f} {terrain[j, i]:.6f}\n")

    f.write(f"\nPOINT_DATA {NX * NY}\n")

    # Élévation
    f.write("SCALARS elevation float 1\n")
    f.write("LOOKUP_TABLE default\n")
    for j in range(NY):
        for i in range(NX):
            f.write(f"{terrain[j, i]:.6f}\n")

    # Masque bâtiments (au sol)
    f.write("SCALARS batiment float 1\n")
    f.write("LOOKUP_TABLE default\n")
    for j in range(NY):
        for i in range(NX):
            f.write(f"{float(building_2d[j, i]):.0f}\n")

print(f"  → {NX}×{NY} points, Z = élévation du terrain")

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT 2 : BÂTIMENTS (polydata — boîtes extrudées sur le terrain)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nExport bâtiments : {args.out_buildings}")

dx = xs[1] - xs[0] if NX > 1 else 1.0
dy = ys[1] - ys[0] if NY > 1 else 1.0

# Collecter les boîtes des bâtiments
all_pts = []
all_quads = []
pt_offset = 0

for j in range(NY):
    for i in range(NX):
        if not building_2d[j, i]:
            continue

        x0, x1 = xs[i] - dx/2, xs[i] + dx/2
        y0, y1 = ys[j] - dy/2, ys[j] + dy/2
        z0 = terrain[j, i]           # base = terrain
        z1 = terrain[j, i] + h_bat   # toit

        # 8 sommets de la boîte
        pts = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),  # base
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),  # toit
        ]
        all_pts.extend(pts)

        p = pt_offset
        # 6 faces (quads)
        all_quads.extend([
            (p+0, p+1, p+2, p+3),  # base
            (p+4, p+5, p+6, p+7),  # toit
            (p+0, p+1, p+5, p+4),  # face avant
            (p+1, p+2, p+6, p+5),  # face droite
            (p+2, p+3, p+7, p+6),  # face arrière
            (p+3, p+0, p+4, p+7),  # face gauche
        ])
        pt_offset += 8

n_bat_pts = len(all_pts)
n_bat_quads = len(all_quads)

with open(args.out_buildings, 'w') as f:
    f.write("# vtk DataFile Version 3.0\n")
    f.write("Batiments 3D\n")
    f.write("ASCII\n")
    f.write("DATASET POLYDATA\n")
    f.write(f"POINTS {n_bat_pts} float\n")
    for p in all_pts:
        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    f.write(f"POLYGONS {n_bat_quads} {n_bat_quads * 5}\n")
    for q in all_quads:
        f.write(f"4 {q[0]} {q[1]} {q[2]} {q[3]}\n")

print(f"  → {n_bat_quads} faces ({building_2d.sum()} bâtiments)")

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT 3 : CHEMIN OPTIMAL (polydata polyligne)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nExport chemin : {args.out_chemin}")

path_pts = centers[path]
n = len(path)

with open(args.out_chemin, 'w') as f:
    f.write("# vtk DataFile Version 3.0\n")
    f.write("Chemin optimal 3D\n")
    f.write("ASCII\n")
    f.write("DATASET POLYDATA\n")
    f.write(f"POINTS {n} float\n")
    for p in path_pts:
        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    f.write(f"LINES 1 {n + 1}\n")
    f.write(f"{n}")
    for idx in range(n):
        f.write(f" {idx}")
    f.write("\n")

    f.write(f"POINT_DATA {n}\n")

    f.write("SCALARS dist_cumulee float 1\n")
    f.write("LOOKUP_TABLE default\n")
    for idx in path:
        f.write(f"{dist[idx]:.6f}\n")

    f.write("SCALARS score_local float 1\n")
    f.write("LOOKUP_TABLE default\n")
    for idx in path:
        f.write(f"{node_score[idx]:.6f}\n")

    f.write("SCALARS altitude float 1\n")
    f.write("LOOKUP_TABLE default\n")
    for idx in path:
        f.write(f"{centers[idx, 2]:.6f}\n")

print(f"  → {n} points, {n-1} segments")

# ── RÉSUMÉ ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Fichiers exportés :")
print(f"  {args.out_terrain}    — terrain (surface 3D)")
print(f"  {args.out_buildings}  — bâtiments (volumes)")
print(f"  {args.out_chemin}     — chemin optimal")
print(f"\nDans ParaView :")
print(f"  1. File → Open → sélectionner les 3 fichiers")
print(f"  2. Apply sur chacun")
print(f"  3. Terrain : colorer par 'elevation', colormap 'terrain'")
print(f"  4. Bâtiments : couleur grise (Solid Color)")
print(f"  5. Chemin : colorer par 'score_local', épaisseur de ligne ≥ 3")
print(f"{'='*60}")

# ── PREVIEW PYVISTA (optionnel) ───────────────────────────────────────────────
if args.preview:
    import pyvista as pv

    print("\nOuverture preview pyvista...")

    terrain_mesh = pv.read(args.out_terrain)
    buildings_mesh = pv.read(args.out_buildings)
    chemin_mesh = pv.read(args.out_chemin)

    plotter = pv.Plotter(window_size=(1200, 800))
    plotter.set_background('#1a1a2e')

    plotter.add_mesh(terrain_mesh, scalars="elevation", cmap="terrain",
                     show_scalar_bar=True,
                     scalar_bar_args={"title": "Élévation (m)", "color": "white"})

    if buildings_mesh.n_cells > 0:
        plotter.add_mesh(buildings_mesh, color="#4a4a6a", opacity=0.7)

    plotter.add_mesh(chemin_mesh, scalars="score_local", cmap="RdYlGn_r",
                     line_width=5,
                     show_scalar_bar=True,
                     scalar_bar_args={"title": "Score danger", "color": "white"})

    pt_A = pv.PolyData(centers[source].reshape(1, 3))
    pt_B = pv.PolyData(centers[target].reshape(1, 3))
    plotter.add_mesh(pt_A, color="#00ff88", point_size=20, render_points_as_spheres=True)
    plotter.add_mesh(pt_B, color="#ff4466", point_size=20, render_points_as_spheres=True)

    plotter.camera.azimuth = 30
    plotter.camera.elevation = 25
    plotter.reset_camera()
    plotter.show()