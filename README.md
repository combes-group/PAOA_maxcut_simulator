# Probabilistic Approximate Optimization Algorithm Simulator with Application to the Maximum Cut CSP.
## Created by Gal Weitz, Department of Applied Mathematics, Department of Physics, University of Colorado Boulder, March 2022.

The Probabilistic Approximate Optimization Algorithm (PAOA) is a classical algorithm that enables the mapping of the
Max-Cut CSP to a probabilistic space, and solve it using an approximation method. Hardware implementation of the PAOA
could be implemented using p-bits as decribed in the Camsari et al. 2019 https://arxiv.org/abs/1809.04028. The goal of
the PAOA is to best replicate the Quantum Approximate Optimization Algorithm (QAOA), first introduced by Farhi et al.
in 2014 https://arxiv.org/abs/1411.4028. By only replacing the quantum component of the QAOA, the solutions of PAOA
serve as a quality control mechagnism of the solutions of the QAOA by direct comparison.
