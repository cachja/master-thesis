# Viscoelastic problems implemented in FEniCS
## Abstract
Describing viscoelastic fluids is a difficult task, as the viscoelastic phenomena are not fully understood. This work follows a method for deriving viscoelastic models that accurately capture the behavior of fluids with polymeric substances, which macroscopically manifest as the stress diffusion, within a consistent thermodynamic framework. We implemented these models using the open-source computing platform FEniCS as a finite element library for Python, and we provide a numerical study of the stress diffusion as a stabilization. By extending our implementation using the arbitrary Lagrangian-Eulerian method, we are able to simulate well-known non-Newtonian phenomena, demonstrating the effectiveness of our approach in enabling a better understanding of these complex fluids.
## Files
### Viscoelastic flow past the cylinder
MeshMaker.py: Generates meshes for symmetrical part of the classical benchmark flow past the cylinder.

Oldroyd-B\_Giesekus\_Combo\_MinLambda.py: Provides minimal $\lambda^*$ for the benchmark of flow past the cylinder using derived vsicoelastic models with the stress diffusion.

Oldroyd-B\_Giesekus\_Combo.py: Provides results for given $\lambda$ for the benchmark of flow past the cylinder using derived vsicoelastic models with the stress diffusion.

### Viscoelastic axisymmetric problems and with a free surface
AxisymmetricShearFlow.py: Solves the axisymmetric shearflow problem for a viscoelastic fluid described by the classical Giesekus model.

ViscoelasticPush\_time-study.py: Solves pressing of a rectangular piece of viscoelastic fluid with a free surface and gives data on volume preservation for different time schemes ((GL) and (BE)) and timesteps.

RodClimbing.py: Solves Rod Climbing (Weissenberg) problem for the classical Oldroyd-B model, but any convex combination of Oldroyd-B and Giesekus model with the stress diffusion is possible.
