# delta_stepping

Pathfinding sur carte de dégâts urbains : calcule, pour une scène 3D avec bâtiments,
le trajet A→B qui **minimise les dégâts cumulés** en cas d'explosion accidentelle
en cours de transport (charge de 40 kg TNT équivalent).

Le "coût" de chaque case de la grille est un score de dégâts obtenu en combinant :
- un modèle analytique **Kingery-Bulmash** (surpression en champ libre) atténué par
  raycasting 2D sur les façades des bâtiments,
- un **Processus Gaussien** entraîné sur des simulations FEA (Abaqus) qui corrige
  le biais du modèle analytique.

Le chemin optimal est ensuite calculé sur le graphe résultant avec l'algorithme
**Delta-Stepping** (version séquentielle et version parallèle MPI).

Le contexte détaillé du format de la carte de dégâts (`damage_map.npz`) est documenté
dans [transmission.md](transmission.md).

## Pipeline

Le pipeline 2D (racine du dépôt) :

| Étape | Script | Entrée | Sortie |
|---|---|---|---|
| 1 | [raycasting_kb.py](raycasting_kb.py) | — | modèle KB + raycasting |
| 2 | [fndegat.py](fndegat.py) | `output-post.vtk` (résultat FEA) | score de dégâts matériels |
| 3 | [gp_damage.py](gp_damage.py) | résultats de simulation | `damage_map.npz`, `damage_map.png` |
| 4 | [precompute_scores.py](precompute_scores.py) | `damage_map.npz` | `graph.npz`, `meta.npz` |
| 5 | [delta_stepping.py](delta_stepping.py) | `graph.npz`, `meta.npz` | `chemin_optimal.vtk` |
| 6 | [visualize_chemin.py](visualize_chemin.py) | `chemin_optimal.vtk`, `meta.npz` | `visualisation_chemin.png` |

[interactive_pathfinder.py](interactive_pathfinder.py) offre une alternative interactive
aux étapes 5-6 : sélection de A/B à la souris sur la carte de danger, calcul et
affichage du chemin en direct.

[job.slurm](job.slurm) est un script de soumission SLURM pour lancer le pipeline sur
un cluster de calcul (`precompute_scores.py` → Delta-Stepping MPI → `visualize_chemin.py`).

Le pipeline 3D ([delta3d/](delta3d/)) reprend la même logique en ajoutant l'altitude
(terrain, hauteur de vol/déplacement, graphe 26-connexe) — voir les docstrings de
chaque script pour le détail des entrées/sorties.

## Installation

```bash
pip install -r requirements.txt
```

`mpi4py` nécessite une implémentation MPI installée sur le système (ex. OpenMPI) pour
les versions parallèles (`delta_stepping_mpi_3d.py`, et `delta_stepping_mpi.py` sur
cluster via `job.slurm`).

## Données

Les fichiers de données générés (`*.npz`, `*.vtk`, `*.vti`) ne sont **pas** versionnés
(voir [.gitignore](.gitignore)) : ils sont volumineux (jusqu'à plus de 100 Mo) et se
régénèrent en relançant le pipeline ci-dessus dans l'ordre. Il faut donc partir de tes
propres résultats de simulation (FEA / scène) pour reproduire une carte de dégâts.

## Limitations connues

- [gp_damage.py](gp_damage.py) importe deux modules non présents dans ce dépôt
  (`from kb import compute_pressure_for_source` et `from dammage import score_from_pressure`) :
  le script ne s'exécute pas tel quel tant que ces modules ne sont pas fournis séparément.
- [fndegat.py](fndegat.py) référence un chemin relatif externe au dépôt
  (`../../03-POST/2D/output-post.vtk`), issu de l'arborescence de simulation d'origine.
- [job.slurm](job.slurm) appelle `delta_stepping_mpi.py`, qui n'est pas présent dans ce
  dépôt (seule la version séquentielle [delta_stepping.py](delta_stepping.py) et la
  version 3D [delta3d/delta_stepping_mpi_3d.py](delta3d/delta_stepping_mpi_3d.py) le sont).
