#ifndef GRID_ALGORITHMS_H
#define GRID_ALGORITHMS_H

#include <algorithms/SparseMatrix.h>
#include <algorithms/LinearOperator.h>
#include <algorithms/Preconditioner.h>

#include <algorithms/approx/Zolotarev.h>
#include <algorithms/approx/Chebyshev.h>
#include <algorithms/approx/Remez.h>
#include <algorithms/approx/MultiShiftFunction.h>

#include <algorithms/iterative/ConjugateGradient.h>
#include <algorithms/iterative/ConjugateResidual.h>
#include <algorithms/iterative/NormalEquations.h>
#include <algorithms/iterative/SchurRedBlack.h>

#include <algorithms/iterative/ConjugateGradientMultiShift.h>

// Lanczos support
#include <algorithms/iterative/MatrixUtils.h>
//#include <algorithms/iterative/ImplicitlyRestartedLanczos.h>

#include <algorithms/CoarsenedMatrix.h>

// Eigen/lanczos
// EigCg
// MCR
// Pcg
// Multishift CG
// Hdcg
// GCR
// etc..

// integrator/Leapfrog
// integrator/Omelyan
// integrator/ForceGradient

// montecarlo/hmc
// montecarlo/rhmc
// montecarlo/metropolis
// etc...


#endif
