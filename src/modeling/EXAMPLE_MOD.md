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

## HC-004-MOD_tensorial_entanglement_curvature.py
Simulates a schematic 3D PEPS network on a 16×16×16 lattice and visualizes local entropy.  
`python HC-004-MOD_tensorial_entanglement_curvature.py`

## HC-005-MOD_effective_field_equation.py
Simulates an effective gravitational field equation derived from quantum entanglement entropy.
`python HC-005-MOD_effective_field_equation.py`

## HC-006-MOD_gravity_laplacian.py
Computes entanglement entropy and its Laplacian (curvature proxy) from a 3D PEPS model.  
`python HC-006-MOD_gravity_laplacian.py --steps 5 --plot`

## HC-007-MOD_gravity_laplacian_impurity.py
Same as HC-006-MOD_gravity_laplacian.py, but introduces a 3×3×3 impurity region with increased transverse field.
`python HC-007-MOD_gravity_laplacian_impurity.py --h-impurity 4.0 --plot`