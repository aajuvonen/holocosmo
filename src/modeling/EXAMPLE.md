# Modeling Scripts — Usage Examples

## gravity_laplacian_simulation.py
Computes entanglement entropy and its Laplacian (curvature proxy) from a 3D PEPS model.  
`python gravity_laplacian_simulation.py --steps 5 --plot`

## gravity_laplacian_impurity.py
Same as gravity_laplacian_simulation.py  , but introduces a 3×3×3 impurity region with increased transverse field.
`python gravity_laplacian_impurity.py --h-impurity 4.0 --plot`
