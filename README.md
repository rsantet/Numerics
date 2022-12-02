RM(G)HMC algorithm for sampling either one-dimensional or multi-dimensional probability measures.
It includes both the Generalized Störmer-Verlet (GSV) and the Implicit Midpoint Rule (IMR) numerical schemes to integrate the Hamiltonian dynamics, relying on Newton's method to solve the implicit problem. A reversal check is used in order for the sampling to be unbiased.
Note that GSV is seen as the composition of Euler B and Euler A numerical schemes, and some redundancy is not eliminated. This implies that the Hessian of both the potential energy function and the diffusion coefficient are needed for the Newton's method.
RMGHMC is the GHMC variant of RMHMC when momenta is being partially refreshed, and can be seen as a discretization of the Langevin dynamics.

## Prerequisite
* It currently uses the [BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl) package.

## Folder composition
* Use the RM(G)HMC methods inside the ```main.jl``` file for one-dimensional sampling, and ```main_nd.jl``` for multi-dimensional sampling.
* See the files ```rmhmc_test.jl``` and ```rmhmc_test_nd.jl``` for examples on how to implement the different functions needed for RMHMC (potential energy function, diffusion coefficient, their derivatives, etc.)

## Improvements in the future:
* Profile
* Stabilize the structures that are used throughout the code (BlockArray, full Array, sparse Array, ...)
* Include the usual implementation of GSV to eliminate the use of both the Hessian of the potential energy function and the diffusion coefficient (only the gradient of the diffusion coefficient is really needed)
* Include the fixed-point iteration method
* Include a mid-point Euler scheme to partially refresh the momenta for the RMGHMC algorithm instead of only relying on an exact analytical integration