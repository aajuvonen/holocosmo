# Modeling Scripts — Usage Examples

### HC-001-MOD_dynamic_holographic_dark_energy.py
Simulates holographic vacuum cosmology using dynamical horizon evolution. Saves derived quantities to CSV to `data/processed` and exports figures to `data/figures/`.
`python HC-001-MOD_dynamic_holographic_dark_energy.py`

## HC-002-MOD_entanglement_curvature_1d.py
Demonstrates curvature tensor extraction from a synthetic entropy field.  
`python HC-002-MOD_entanglement_curvature_1d.py`

## HC-003-MOD_entanglement_curvature_3d.py
Simulates a 2x2x2 spin lattice and extracts entanglement curvature via entropy Laplacian.  
`python HC-003-MOD_entanglement_curvature_3d.py`

## entanglement_peps_3d_demo.py
Simulates a schematic 3D PEPS network on a 16×16×16 lattice and visualizes local entropy.  
`python entanglement_peps_3d_demo.py`

## gravity_laplacian_simulation.py
Computes entanglement entropy and its Laplacian (curvature proxy) from a 3D PEPS model.  
`python gravity_laplacian_simulation.py --steps 5 --plot`

## gravity_laplacian_impurity.py
Same as gravity_laplacian_simulation.py  , but introduces a 3×3×3 impurity region with increased transverse field.
`python gravity_laplacian_impurity.py --h-impurity 4.0 --plot`