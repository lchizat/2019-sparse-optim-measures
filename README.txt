This is the julia code for the paper

Sparse Optimization on Measures with Over-parameterized Gradient Descent
Lenaic Chizat, 2019.

In order to reproduce the experiments, you need to install julia 1.0 (or later) and the following packages: PyPlot, ProgressMeter, Random, LinearAlgebra. Then in julia prompt run include("filename.jl")

The running times for a standard laptop are:

illustration_generic (10 seconds)
illustration_reluNN (4 seconds)
illustration_deconv1D (5 seconds)
illustration_deconv2D (5 minutes)

cvgce_deconv2D (13 minutes)
cvgce_2NNalt (30 minutes)

success_deconv1D (2 minutes)
success_2NNalt (40 minutes)

vertical_compare (13 minutes)



