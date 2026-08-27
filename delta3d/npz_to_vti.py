"""
npz_to_vti.py
Convertit damage_map_3d.npz en .vti pour ParaView.

Usage :
    python npz_to_vti.py
    python npz_to_vti.py --input damage_map_3d.npz --output damage_map_3d.vti
"""

import argparse
import numpy as np
import meshio

parser = argparse.ArgumentParser(description="Conversion NPZ → VTI pour ParaView")
parser.add_argument("--input",  default="damage_map_3d.npz", help="Carte 3D npz")
parser.add_argument("--output", default="damage_map_3d.vti", help="Sortie VTI")
args = parser.parse_args()

print(f"Chargement : {args.input}")
d = np.load(args.input, allow_pickle=True)

xs = d['xs']
ys = d['ys']
zs = d['zs']
damage   = d['damage_map_3d']    # (NZ, NY, NX)
building = d['building_mask_3d'] # (NZ, NY, NX)

NX, NY, NZ = len(xs), len(ys), len(zs)
print(f"Grille : {NX}×{NY}×{NZ}")

# VTI = grille régulière, on a besoin de l'origine et du spacing
dx = xs[1] - xs[0] if NX > 1 else 1.0
dy = ys[1] - ys[0] if NY > 1 else 1.0
dz = zs[1] - zs[0] if NZ > 1 else 1.0

# Écriture via VTK directement (plus fiable que meshio pour ImageData)
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    grid = vtk.vtkImageData()
    grid.SetDimensions(NX + 1, NY + 1, NZ + 1)  # points = cells + 1
    grid.SetOrigin(xs[0], ys[0], zs[0])
    grid.SetSpacing(dx, dy, dz)

    # Danger — ordre Fortran (VTK attend x-fastest)
    danger_flat = damage.transpose(2, 1, 0).ravel().astype(np.float64)
    # Remplacer inf par -1 pour ParaView
    danger_flat[np.isinf(danger_flat)] = -1.0
    arr1 = numpy_to_vtk(danger_flat, deep=True)
    arr1.SetName("danger")
    grid.GetCellData().AddArray(arr1)

    # Masque bâtiments
    bat_flat = building.transpose(2, 1, 0).ravel().astype(np.float64)
    arr2 = numpy_to_vtk(bat_flat, deep=True)
    arr2.SetName("batiment")
    grid.GetCellData().AddArray(arr2)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(args.output)
    writer.SetInputData(grid)
    writer.Write()

    print(f"Exporté : {args.output}")
    print(f"  Ouvre avec ParaView, applique un 'Threshold' sur 'batiment' pour voir les bâtiments")
    print(f"  et un 'Volume Rendering' sur 'danger' pour le champ de dégâts")

except ImportError:
    # Fallback sans VTK : export en VTR via meshio (moins propre mais fonctionnel)
    print("VTK non disponible, export en format brut...")

    # Alternative : sauvegarder en .npy séparés + un script ParaView
    # Ou utiliser meshio avec des points explicites
    from itertools import product

    print("Fallback : export CSV pour ParaView (Table To Structured Grid)")
    output_csv = args.output.replace('.vti', '.csv')

    with open(output_csv, 'w') as f:
        f.write("x,y,z,danger,batiment\n")
        for k in range(NZ):
            for j in range(NY):
                for i in range(NX):
                    val = damage[k, j, i]
                    if np.isinf(val):
                        val = -1.0
                    bat = int(building[k, j, i])
                    f.write(f"{xs[i]:.3f},{ys[j]:.3f},{zs[k]:.3f},{val:.6f},{bat}\n")

    print(f"Exporté : {output_csv}")
    print(f"  Dans ParaView : Filters → Table To Structured Grid")