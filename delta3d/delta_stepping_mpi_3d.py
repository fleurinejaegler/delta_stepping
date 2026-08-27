"""
delta_stepping_mpi_3d.py
Delta-Stepping parallèle (MPI) — graphe de dégâts urbains 3D.

Mode véhicule (--z_above_ground) :
  Le véhicule suit le terrain à une hauteur constante au-dessus du sol.
  Pour chaque cellule (i,j), on sélectionne la couche k telle que
  zs[k] ≈ terrain(i,j) + z_above_ground.
  Le graphe est réduit à cette surface dans l'espace 3D.

Mode libre (par défaut) :
  Chemin libre dans tout le volume 3D.

Usage :
    mpirun -np 4 python3 delta_stepping_mpi_3d.py
    mpirun -np 4 python3 delta_stepping_mpi_3d.py --z_above_ground 1.0
    mpirun -np 4 python3 delta_stepping_mpi_3d.py --z_above_ground 1.0 20 20 80 80
"""

import sys
import argparse
import numpy as np
from collections import defaultdict
from mpi4py import MPI
from scipy.spatial import cKDTree

# ── MPI INIT ──────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def log(msg):
    if rank == 0:
        print(msg, flush=True)

# ── ARGUMENTS ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Delta-Stepping MPI 3D")
parser.add_argument("--graph",           default="graph_3d.npz")
parser.add_argument("--meta",            default="meta_3d.npz")
parser.add_argument("--damage",          default="damage_map_3d.npz")
parser.add_argument("--out_vtk",         default="chemin_optimal_3d.vtk")
parser.add_argument("--out_npz",         default="result_3d.npz")
parser.add_argument("--z_above_ground",  type=float, default=None,
                    help="Hauteur du véhicule au-dessus du sol (ex: 1.0 m). "
                         "Nécessite terrain_elevation dans damage_map_3d.npz.")
parser.add_argument("coords", nargs='*', type=float,
                    help="Ax Ay Bx By (mode véhicule) ou Ax Ay Az Bx By Bz (mode libre)")
args = parser.parse_args()

# ── CHARGEMENT ────────────────────────────────────────────────────────────────
log("Chargement du graphe 3D...")
g    = np.load(args.graph)
meta = np.load(args.meta, allow_pickle=True)

n_nodes_total = int(g['n_nodes'][0])
rows       = g['rows']
cols       = g['cols']
weights    = g['weights']
centers    = meta['centers']
node_score = meta['node_score']

xs = meta['xs']
ys = meta['ys']
zs = meta['zs']
NX = int(meta['NX'][0])
NY = int(meta['NY'][0])
NZ = int(meta['NZ'][0])

log(f"Graphe 3D complet : {n_nodes_total:,} nœuds  {len(rows):,} arêtes")
log(f"Grille : {NX}×{NY}×{NZ}")

# ── MODE VÉHICULE : SURFACE TERRAIN + OFFSET ─────────────────────────────────
valid_nodes = None
terrain = None

if args.z_above_ground is not None:
    z_offset = args.z_above_ground

    dmg = np.load(args.damage, allow_pickle=True)
    if 'terrain_elevation' not in dmg:
        log("ERREUR : terrain_elevation absent de damage_map_3d.npz")
        log("Lance d'abord : python3 interpolate_terrain.py")
        sys.exit(1)

    terrain = dmg['terrain_elevation']  # (NY, NX)
    dmg.close()

    log(f"\nMode véhicule : {z_offset:.1f}m au-dessus du sol")
    log(f"  Terrain : min={terrain.min():.2f}m  max={terrain.max():.2f}m")

    # Pour chaque cellule (i,j), trouver la couche k telle que
    # zs[k] ≈ terrain(j,i) + z_offset
    valid_node_list = []
    vehicle_z_map = np.zeros((NY, NX), dtype=np.float64)

    for j in range(NY):
        for i in range(NX):
            z_target = terrain[j, i] + z_offset
            k_best = int(np.argmin(np.abs(zs - z_target)))
            k_best = np.clip(k_best, 0, NZ - 1)
            node_id = k_best * (NX * NY) + j * NX + i
            valid_node_list.append(node_id)
            vehicle_z_map[j, i] = zs[k_best]

    valid_nodes = set(valid_node_list)

    log(f"  Nœuds sur la surface véhicule : {len(valid_nodes):,}")
    log(f"  Z véhicule : min={vehicle_z_map.min():.2f}m  max={vehicle_z_map.max():.2f}m")

    # Filtrer les arêtes (vectorisé)
    valid_mask = np.zeros(n_nodes_total, dtype=bool)
    valid_mask[valid_node_list] = True

    edge_mask = valid_mask[rows] & valid_mask[cols]
    rows    = rows[edge_mask]
    cols    = cols[edge_mask]
    weights = weights[edge_mask]

    log(f"  Arêtes retenues : {len(rows):,}")
else:
    log("\nMode libre : chemin 3D complet")

# ── GRAPHE D'ADJACENCE ───────────────────────────────────────────────────────
n_nodes = n_nodes_total
adj = defaultdict(list)
for u, v, w in zip(rows, cols, weights):
    adj[int(u)].append((int(v), float(w)))

log(f"Processus MPI : {size}")

# ── PARTITIONNEMENT ──────────────────────────────────────────────────────────
if valid_nodes is not None:
    # Mode véhicule : répartition équitable des nœuds valides
    valid_sorted = np.array(sorted(valid_nodes), dtype=np.int64)
    parts = np.array_split(valid_sorted, size)

    node_owner = np.full(n_nodes, -1, dtype=np.int32)
    for r, part in enumerate(parts):
        node_owner[part] = r

    my_nodes = set(parts[rank].tolist())
    log(f"  Rang {rank} : {len(my_nodes):,} nœuds")
else:
    # Mode libre : slabs en Z
    slabs = np.array_split(np.arange(NZ), size)
    node_owner = np.zeros(n_nodes, dtype=np.int32)
    for r, slab in enumerate(slabs):
        for k in slab:
            start = k * NX * NY
            end   = start + NX * NY
            node_owner[start:end] = r
    my_nodes = set(np.where(node_owner == rank)[0].tolist())

# ── CALIBRATION DELTA ─────────────────────────────────────────────────────────
def calibrer_delta(adj, percentile=25):
    all_w = [w for u in adj for _, w in adj[u] if w > 0 and not np.isinf(w)]
    if not all_w:
        return 1.0
    return float(np.percentile(all_w, percentile))

delta = calibrer_delta(adj)
log(f"Delta (p25) : {delta:.4f}")

# ── PARSE COORDONNÉES ────────────────────────────────────────────────────────
coords = args.coords

if args.z_above_ground is not None:
    # Mode véhicule : Ax Ay Bx By — Z déterminé par le terrain
    if len(coords) == 4:
        ax, ay, bx, by = coords
    else:
        ax, ay = 20.0, 20.0
        bx, by = 80.0, 80.0

    from scipy.interpolate import RegularGridInterpolator
    terp = RegularGridInterpolator((ys, xs), terrain, method='nearest',
                                    bounds_error=False, fill_value=None)
    z_a = float(terp([ay, ax])[0]) + args.z_above_ground
    z_b = float(terp([by, bx])[0]) + args.z_above_ground

    A_XY = np.array([ax, ay, z_a])
    B_XY = np.array([bx, by, z_b])

    log(f"Terrain sous A : {z_a - args.z_above_ground:.2f}m → véhicule à z={z_a:.2f}m")
    log(f"Terrain sous B : {z_b - args.z_above_ground:.2f}m → véhicule à z={z_b:.2f}m")
else:
    # Mode libre : Ax Ay Az Bx By Bz
    if len(coords) == 6:
        A_XY = np.array(coords[:3])
        B_XY = np.array(coords[3:])
    else:
        A_XY = np.array([20.0, 20.0, 0.0])
        B_XY = np.array([80.0, 80.0, 25.0])

# Snap vers nœuds valides
if valid_nodes is not None:
    valid_list = np.array(sorted(valid_nodes))
    valid_centers = centers[valid_list]
    tree_valid = cKDTree(valid_centers)
    source = valid_list[int(tree_valid.query(A_XY)[1])]
    target = valid_list[int(tree_valid.query(B_XY)[1])]
else:
    tree = cKDTree(centers)
    source = int(tree.query(A_XY)[1])
    target = int(tree.query(B_XY)[1])

log(f"\nSource A : nœud {source}  ({centers[source, 0]:.1f}, {centers[source, 1]:.1f}, {centers[source, 2]:.1f})")
log(f"Cible  B : nœud {target}  ({centers[target, 0]:.1f}, {centers[target, 1]:.1f}, {centers[target, 2]:.1f})")

# ── DELTA-STEPPING MPI ────────────────────────────────────────────────────────
def delta_stepping_mpi(source, target):
    INF = float("inf")
    dist = np.full(n_nodes, INF, dtype=np.float64)
    pred = np.full(n_nodes, -1,  dtype=np.int32)
    dist[source] = 0.0

    def bucket_idx(d):
        return int(d / delta)

    buckets = defaultdict(set)
    if source in my_nodes:
        buckets[0].add(source)

    outgoing = defaultdict(list)
    iteration = 0
    max_iter  = n_nodes * 2

    while iteration < max_iter:
        iteration += 1

        # Plus petit bucket actif global
        local_min = min((k for k, b in buckets.items() if b), default=n_nodes + 1)
        global_min = comm.allreduce(local_min, op=MPI.MIN)

        if global_min > n_nodes:
            break

        # Early termination
        global_dist_target = comm.allreduce(dist[target], op=MPI.MIN)
        if global_dist_target < INF and global_min > bucket_idx(global_dist_target):
            break

        i = global_min
        R = set()

        # Phase légère
        while buckets.get(i, set()):
            S = set(buckets[i])
            buckets[i].clear()
            R.update(S)

            for u in S:
                for v, w in adj[u]:
                    if w <= delta:
                        if np.isinf(w):
                            continue
                        new_d = dist[u] + w
                        if new_d < dist[v]:
                            old_d = dist[v]
                            dist[v] = new_d
                            pred[v] = u
                            if node_owner[v] == rank:
                                if old_d < INF:
                                    buckets[bucket_idx(old_d)].discard(v)
                                buckets[bucket_idx(new_d)].add(v)
                            else:
                                outgoing[node_owner[v]].append((v, new_d, u))

        # Phase lourde
        for u in R:
            for v, w in adj[u]:
                if w > delta and not np.isinf(w):
                    new_d = dist[u] + w
                    if new_d < dist[v]:
                        old_d = dist[v]
                        dist[v] = new_d
                        pred[v] = u
                        if node_owner[v] == rank:
                            if old_d < INF:
                                buckets[bucket_idx(old_d)].discard(v)
                            buckets[bucket_idx(new_d)].add(v)
                        else:
                            outgoing[node_owner[v]].append((v, new_d, u))

        # ── SYNCHRONISATION MPI ──────────────────────────────────────────
        # Échange collectif des relaxations
        send_bufs = [outgoing.get(r, []) for r in range(size)]
        recv_bufs = comm.alltoall(send_bufs)

        for buf in recv_bufs:
            for v, new_d, pred_u in buf:
                if new_d < dist[v]:
                    old_d = dist[v]
                    dist[v] = new_d
                    pred[v] = pred_u
                    if old_d < INF:
                        buckets[bucket_idx(old_d)].discard(v)
                    buckets[bucket_idx(new_d)].add(v)

        outgoing.clear()

        # Allreduce sur dist pour cohérence globale
        all_dist = np.empty_like(dist)
        comm.Allreduce(dist, all_dist, op=MPI.MIN)
        dist[:] = all_dist

    log(f"  Convergé en {iteration} itérations")

    # Synchronisation finale de pred vers tous les processus
    # Chaque propriétaire partage ses valeurs ; sentinelle -2 pour les non-propriétaires
    pred_local = np.full(n_nodes, -2, dtype=np.int32)
    mask_owner = node_owner == rank
    pred_local[mask_owner] = pred[mask_owner]
    pred_synced = np.empty(n_nodes, dtype=np.int32)
    comm.Allreduce(pred_local, pred_synced, op=MPI.MAX)
    pred[:] = pred_synced

    return dist, pred


# ── EXÉCUTION ─────────────────────────────────────────────────────────────────
log(f"\nDelta-Stepping MPI 3D (delta={delta:.4f}, {size} procs)...")
dist, pred = delta_stepping_mpi(source, target)

# ── RECONSTRUCTION CHEMIN (rang 0) ───────────────────────────────────────────
if rank == 0:
    path = []
    if pred[target] >= 0 or target == source:
        cur = target
        visited = set()
        while cur != source and cur not in visited:
            visited.add(cur)
            path.append(cur)
            cur = pred[cur]
        if cur == source:
            path.append(source)
            path.reverse()
        else:
            path = []

    if path:
        print(f"\nDanger cumulé total  : {dist[target]:.4f}")
        print(f"Longueur du chemin   : {len(path)} nœuds")
        print(f"Danger moyen/nœud    : {dist[target]/len(path):.4f}")

        if args.z_above_ground is not None:
            path_z = centers[path, 2]
            print(f"Altitude véhicule    : min={path_z.min():.2f}m  max={path_z.max():.2f}m")

        # Export résultats NPZ
        save_dict = dict(
            path=np.array(path),
            dist=dist,
            pred=pred,
            source=np.array([source]),
            target=np.array([target]),
        )
        if args.z_above_ground is not None:
            save_dict['z_above_ground'] = np.array([args.z_above_ground])
            save_dict['vehicle_mode'] = np.array([True])
        np.savez(args.out_npz, **save_dict)
        print(f"Résultats : {args.out_npz}")

        # Export VTK ASCII (sans dépendance pyvista)
        try:
            n   = len(path)
            pts = centers[path]
            with open(args.out_vtk, 'w') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write("Chemin optimal 3D - vehicule\n")
                f.write("ASCII\n")
                f.write("DATASET POLYDATA\n")
                f.write(f"POINTS {n} float\n")
                for p in pts:
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
            print(f"VTK : {args.out_vtk}")
        except Exception as e:
            print(f"Export VTK échoué : {e}")
    else:
        print("\nAucun chemin trouvé entre A et B.")
        print("Essaie d'autres coordonnées.")
