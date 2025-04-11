# Analysis Scripts — Usage Examples

## entanglement_cluster_analysis.py
Clusters and correlates entanglement curvature from CSV input.  
`python entanglement_cluster_analysis.py --input peps_results.csv --output-dir results/`

## entanglement_curvature_gradient.py
Visualizes the gradient of curvature (∇Laplacian) on a fixed z-slice from PEPS results.  
`python entanglement_curvature_gradient.py --input peps_results.csv --slice-z 16`

## entanglement_curvature_geodesic.py  
Traces a geodesic path through a 3D curvature field using Dijkstra's algorithm.  
`python entanglement_curvature_geodesic.py --input cluster_analysis.csv --start 2 2 2 --end 29 29 29`

## entanglement_curvature_deviation.py  
Measures how small changes in geodesic start points impact path trajectories through curvature.  
`python entanglement_curvature_deviation.py --input cluster_analysis.csv --start 2 2 2 --end 29 29 29`

## sparc_fitting.py  
Fits SPARC-format galaxy rotation curves using an entropic gravity model.  
`python sparc_fitting.py`
