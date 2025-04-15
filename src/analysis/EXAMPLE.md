# Analysis Scripts — Usage Examples

## HC-008-ANA_cluster_analysis.py
Clusters and correlates entanglement curvature from CSV input of HC-006-MOD or HC-007-MOD.
`python HC-008-ANA_cluster_analysis.py --input peps_results.csv --output-dir results/`

## HC-009-ANA_curvature deviation.py
Measures from HC-008-ANA output changes in geodesic start points impact path trajectories through curvature.
`python HC-009-ANA_curvature deviation.py --input cluster_analysis.csv --start 2 2 2 --end 29 29 29`

## HC-010-ANA_curvature_geodesic.py  
Traces a geodesic path through a 3D curvature field of HC-008-ANA output using Dijkstra's algorithm.
`python HC-010-ANA_curvature_geodesic.py --input cluster_analysis.csv --start 2 2 2 --end 29 29 29`

## HC-011-ANA_curvature_gradient.py
Visualizes the gradient of curvature (∇Laplacian) on a fixed z-slice from PEPS results of HC-006-MOD or HC-007-MOD.
`python HC-011-ANA_curvature_gradient.py --input peps_results.csv --slice-z 16`

## HC-012-ANA_visualize_laplacian_3d.py  
Generates a 3D scatter plot of Laplacian (entanglement curvature) values from a PEPS simulation cluster analysis of HC-008-ANA.
`python HC-012-ANA_visualize_laplacian_3d.py --input cluster_analysis.csv --threshold 0.025 --sample 3000 --show`

## HC-013-ANA_sparc_fitting.py  
Fits SPARC-format galaxy rotation curves using an entropic gravity model.  
`python HC-013-ANA_sparc_fitting.py`

## HC-014-ANA_radial_profile.py
Computes radial averages of the curvature (Laplacian) field from a 3D entanglement lattice of HC-008-ANA.
`python radial_profile.py --input-file data.csv --output-file radial_profile.csv --bins 50`

## HC-015-ANA_potential_fitting.py
Fits an effective gravitational potential (Yukawa or Gaussian) to radial curvature data extracted from PEPS simulations of ouput from HC-014-ANA.
Outputs a figure and a CSV with best-fit parameters and residuals.  
`python potential_fitting.py --input-file radial_profile.csv --model yukawa --output-file potential_fit.svg`