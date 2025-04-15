# HoloCosmo Modeling & Analysis Workflow

This diagram shows the end-to-end dependency flow between simulation (MOD) scripts and analysis (ANA) scripts in the HoloCosmo project.

- **MOD scripts** generate raw PEPS-based entanglement curvature data.
- **ANA scripts** perform clustering, curvature visualizations, geodesic tracing, radial profiling, and potential fitting.

```mermaid
flowchart TD

  subgraph "Modeling (MOD)"
    HC006[HC-006-MOD<br/>Gravity Laplacian]
    HC007[HC-007-MOD<br/>Gravity Laplacian impurity]
  end

  subgraph "Analysis (ANA)"
    HC008[HC-008-ANA<br/>Cluster analysis]
    HC009[HC-009-ANA<br/>Curvature deviation]
    HC010[HC-010-ANA<br/>Curvature geodesic]
    HC011[HC-011-ANA<br/>Curvature gradient]
    HC012[HC-012-ANA<br/>Laplacian_33]
    HC014[HC-014-ANA<br/>Radial profile]
    HC015[HC-015-ANA<br/>Potential fitting]
    HC013[HC-013-ANA<br/>Sparc fitting]
  end

  %% MOD feeds ANA
  HC006 --> HC008
  HC007 --> HC008

  HC006 --> HC011
  HC007 --> HC011

  %% Cluster feeds further analysis
  HC008 --> HC009
  HC008 --> HC010
  HC008 --> HC012
  HC008 --> HC014

  HC014 --> HC015
```
