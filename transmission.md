# Carte de dégâts — transmission pour le pathfinding

## Contexte

On cherche le chemin optimal pour transporter une bombe de A vers B dans une scène urbaine de 100×100m,
en minimisant les dégâts en cas d'explosion accidentelle.

La carte ci-jointe (`damage_map.npz` / `damage_map.vtk`) représente, pour chaque position possible
de la bombe dans la scène, le **score de dégâts** qu'une explosion provoquerait à cet endroit.

Ce score a été construit en deux étapes :
1. **Modèle analytique KB** (Kingery-Bulmash) : estimation rapide de la surpression en champ libre,
   avec atténuation par raycasting 2D sur les façades des bâtiments.
2. **Processus Gaussien** entraîné sur 10 simulations FEA (Abaqus) : corrige le biais du modèle KB
   en apprenant le résidu (réel − KB) aux points simulés.

## Fichiers

| Fichier | Format | Contenu |
|---------|--------|---------|
| `damage_map.npz` | NumPy archive | grille + score + incertitude + masque bâtiments |
| `damage_map.vtk` | VTK StructuredGrid | même données, lisible ParaView / pyvista |

## Utilisation en Python (`damage_map.npz`)

```python
import numpy as np

d = np.load('damage_map.npz')

xs            = d['xs']             # (100,) coordonnées x en mètres [1..99]
ys            = d['ys']             # (100,) coordonnées y en mètres [1..99]
damage_map    = d['damage_map']     # (100,100) score de dégâts GP
std_map       = d['std_map']        # (100,100) incertitude σ du GP
building_mask = d['building_mask']  # (100,100) bool — True = bâtiment impassable
```

**Convention d'indexation** : `damage_map[i, j]` correspond à la position `(xs[j], ys[i])`,
soit `x = xs[j]`, `y = ys[i]` (axe 0 = y, axe 1 = x, cohérent avec `np.meshgrid`).

## Construction du coût pour Dijkstra

```python
cost = damage_map.copy()
cost[building_mask] = np.inf   # bâtiments impassables

# Convertir des coordonnées métriques (x, y) en indices grille (i, j) :
def xy_to_ij(x, y):
    j = int(np.argmin(np.abs(xs - x)))
    i = int(np.argmin(np.abs(ys - y)))
    return i, j

start = xy_to_ij(10, 10)   # point A
end   = xy_to_ij(90, 90)   # point B
# → lancer Dijkstra minimax sur cost avec start et end
```

## Interprétation du score

| Score | Signification |
|-------|--------------|
| 0     | Aucun dégât |
| ~3    | Dégâts légers sur bâtiments éloignés |
| ~6    | Dégâts graves sur plusieurs bâtiments |
| ~7.5  | Score max observé (centre de scène, entouré de bâtiments) |

Le score est la somme, sur les 5 bâtiments, du score ICPE moyen pondéré par catégorie :
- Cat 1 (poids 1.0), Cat 2 (poids 2.0), Cat 3 (poids 3.0), Cat 4 (poids 5.0)
- Formule par sonde : `log(clip(p, 20, 2000) / 20) / log(2000/20) × poids`
- Score max théorique = 2+5+1+3+1 = **12**

## Qualité de la carte

LOOCV sur les 10 simulations : **MAE = 0.32**, **RMSE = 0.39** sur une plage [3.6, 7.5].

`std_map` donne l'incertitude du GP en chaque point — les zones avec σ élevé sont peu couvertes
par les simulations, le score y est moins fiable. Il est possible d'en tenir compte dans le
pathfinding (ex. pénaliser les zones à forte incertitude).

## Scène

- 5 bâtiments, catégories ICPE 1 à 4
- Scène 100×100m, grille 1m de résolution (~1m/cellule)
- Charge explosive : 40 kg TNT équivalent
