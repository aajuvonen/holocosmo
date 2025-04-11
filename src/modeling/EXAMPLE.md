# Modeling Scripts — Usage Examples

### holographic_model.py
Simulates holographic vacuum cosmology using dynamical horizon evolution. Saves derived quantities to CSV to `data/processed` and exports figures to `data/figures/`.
`python src/analysis/holographic_model.py`

## entanglement_curvature_tensor_demo.py  
Demonstrates curvature tensor extraction from a synthetic entropy field.  
`python entanglement_curvature_tensor_demo.py`

## entanglement_curvature_tensor_3d_demo.py  
Simulates a 2x2x2 spin lattice and extracts entanglement curvature via entropy Laplacian.  
`python entanglement_curvature_tensor_3d_demo.py`

## entanglement_peps_3d_demo.py  
Simulates a schematic 3D PEPS network on a 16×16×16 lattice and visualizes local entropy.  
`python entanglement_peps_3d_demo.py`

## gravity_laplacian_simulation.py
Computes entanglement entropy and its Laplacian (curvature proxy) from a 3D PEPS model.  
`python gravity_laplacian_simulation.py --steps 5 --plot`

## gravity_laplacian_impurity.py
Same as gravity_laplacian_simulation.py  , but introduces a 3×3×3 impurity region with increased transverse field.
`python gravity_laplacian_impurity.py --h-impurity 4.0 --plot`