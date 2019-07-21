using PyPlot, ProgressMeter
using Random, LinearAlgebra


Random.seed!(1) 
nf = 5 # number of frequency components
coeffs = randn(2nf+1) .+ im .* randn(2nf+1)
phi(x) = real(sum(coeffs[k+nf+1] * exp(im*k*x) for k in -nf:nf))
phi_der(x) = real(sum(coeffs[k+nf+1] * im * k * exp(im*k*x) for k in -nf:nf))


function PGD_scalar(r_init, θ_init, lambda, alpha, beta, niter)
    m = length(r_init)
    rs = zeros(m,niter)
    θs = zeros(m,niter)
    gradr, gradθ = zeros(m), zeros(m)
    r, θ = r_init, θ_init
    loss = zeros(niter)
    for iter = 1:niter
        rs[:,iter] = r
        θs[:,iter] = θ
        ff = sum(r.^2 .* phi.(θ))/m
        loss[iter] = (1/2)*(2+ff)^2 + lambda * sum(r.^2)/m
        gradr = 2r .* ((2+ff) .* phi.(θ) .+ lambda)
        gradθ = (2+ff) .* phi_der.(θ)
        r = r .- alpha * gradr
        θ = θ .- beta * gradθ
    end
    return rs, θs, loss
end

alpha, beta = 0.01, 0.002 #(beta in 0.00001, 0.0001)
lambda =  1.5
m = 30
r_init = 0.5*ones(m)
θ_init = range(0,2π*(1-1/m),length=m)
niter = 3*10^4
rs, θs, loss = PGD_scalar(r_init, θ_init, lambda, alpha, beta, niter)

figure(figsize=[4,3])
xs = range(0,2pi,length=200)
plot(xs,phi.(xs)/3,label=L"\phi","C1")
plot(θs',rs',"k",linewidth=0.8)
plot((θs .- 2π)',rs',"k",linewidth=0.8)
plot((θs .+ 2π)',rs',"k",linewidth=0.8)
plot(θs',rs',"k",linewidth=0.8)
ax = plot(mod.(θs[:,end],2π),rs[:,end],".C3",markersize=8,label="limit of the flow")
axis([0,2π,-2.,2.2])
hlines(0,0,2π,"k")

xlabel(L"\Theta")
#ylabel(L"\mathbb{R}")
xticks([])
yticks([0])
#legend();
savefig("illustration_generic.pdf",bbox_inches="tight")