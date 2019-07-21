using PyPlot, ProgressMeter
using Random, LinearAlgebra

### I. 1D experiment

nf = 2 # number of frequency components
phi(x) = real(sum(exp.(im*k*x) for k in -nf:nf))
phi_der(x) = real(sum(im*k*exp.(im*k*x) for k in -nf:nf))

function PGD_1D_convex(r_init, θ_init, r_obs, θ_obs, lambda, alpha, beta, niter; mirror=true)
    ss(u) = sign(u) * u
    m0 = length(r_obs)
    m = length(r_init)
    rs = zeros(m,niter)
    θs = zeros(m,niter)
    gradr, gradθ = zeros(m), zeros(m)
    r, θ = r_init, θ_init
    loss = zeros(niter)
    @showprogress 1 "Computing..." for iter = 1:niter
        rs[:,iter] = r
        θs[:,iter] = θ
        Kxx = phi.(θ .- θ')
        Kyy = phi.(θ_obs .- θ_obs')
        Kxy = phi.(θ .- θ_obs')
        if mirror
            a = ss.(r)/m
            b = ss.(r_obs)
        else
            a = r/m
            b = r_obs
        end
        loss[iter] = (a' * Kxx * a + b' * Kyy * b - 2a' * Kxy * b)/2 + lambda * sum(abs.(a))
        gradJ = Kxx * a .- Kxy * b 
        if mirror
            r = r .* exp.(- 2 * alpha * (gradJ .+ lambda)) # mirror descent
        else
            r = r .- alpha * gradJ # forward
            r = sign.(r) .* max.(0.0, abs.(r) .- alpha * lambda) # backward
        end
    end
    return rs, θs, loss
end
        
Random.seed!(1) # randomness seed # best was 2,4
m0 = 1 # number of spikes ground truth
# generate random spikes on the domain [0,2π]
θ0 = [π]
w0  = [1]

alpha, beta = 3, 0.0
lambda =  0.3
r_obs = sign.(w0) .* sqrt.(abs.(w0))
θ_obs = θ0
m = 200#0
r_init = 1.2*ones(m)
#r_init[isodd.(1:m)] *= -1.0
θ_init = range(0,2π*(1-1/m),length=m)
niter = 3000
alpha, beta = 0.1, 0.0
println("Mirror descent 1D:")
rsA, θsA, lossA = PGD_1D_convex(r_init, θ_init, r_obs, θ_obs, lambda, alpha, beta, niter, mirror=true)
r_init = 0.0*ones(m)
println("Euclidean descent (ISTA) 1D:")
rsB, θsB, lossB = PGD_1D_convex(r_init.^2, θ_init, r_obs.^2, θ_obs, lambda, alpha, beta, niter, mirror=false)

figure(figsize=[4,3])
loglog(lossA .- lossA[end],label="mirror descent")
loglog(lossB .- lossA[end],label="euclidean descent")
axis([1,1e3,5e-3,5])
xlabel(L"iteration index $k$")
ylabel(L"objective $J(\nu_k)$")
grid("on")
legend()
savefig("vertical_deconv1D.pdf",bbox_inches="tight")
        
        
figure(figsize=[4,3])
iter = 1000
subplot(121)
rs =rsA
θs =θsA
xs = range(0, 2π, length=200)
plot(θs[:,iter],rs[:,iter],".C3",markersize=8,label="limit of the flow")
axis([0,2π,-1.0,30])
hlines(0,0,2π,"k")
xlabel(L"\Theta")
xticks([])
yticks([0])
title("Mirror descent")

subplot(122)
rs =rsB
θs =θsB
xs = range(0, 2π, length=200)
plot(θs[:,iter],rs[:,iter],".C3",markersize=8,label="limit of the flow")
axis([0,2π,-1.0,30])
hlines(0,0,2π,"k")
xlabel(L"\Theta")
xticks([])
yticks([0])
title("Euclidean descent")
savefig("vertical_deconv1D_iterate.pdf",bbox_inches="tight")
        
        
### II. 2D experiment
nf = 2 # number of frequency components
phi(x,y) = real(sum(exp(im*(kx*x + ky*y)) for kx in -nf:nf, ky in -nf:nf))
        
function PGD_2D_convex(r_init, θ_init, r_obs, θ_obs, lambda, alpha, beta, niter; mirror=true)
    ss(r) = abs(r)*r # signed square function
    m0 = length(r_obs)
    m = length(r_init)
    rs = zeros(m, niter)
    θs = zeros(m, 2, niter)
    gradr, gradθ = zeros(m), zeros(m, 2)
    r, θ = r_init, θ_init
    loss = zeros(niter)
    @showprogress 1 "Computing..." for iter = 1:niter
        rs[:,iter] = r
        θs[:,:,iter] = θ
        Kxx = phi.(θ[:,1] .- θ[:,1]',θ[:,2] .- θ[:,2]')
        Kyy = phi.(θ_obs[:,1] .- θ_obs[:,1]',θ_obs[:,2] .- θ_obs[:,2]')
        Kxy = phi.(θ[:,1] .- θ_obs[:,1]', θ[:,2] .- θ_obs[:,2]')
        if mirror
            a = ss.(r)/m
            b = ss.(r_obs)
        else
            a = r/m
            b = r_obs
        end
        loss[iter] = (a' * Kxx * a + b' * Kyy * b - 2a' * Kxy * b)/2 + lambda * sum(abs.(a))
        gradJ = Kxx * a .- Kxy * b
        if mirror
            r = r .* exp.(- 2 * alpha * (gradJ .+ lambda)) # mirror descent
        else
            r = r .- alpha * gradJ # forward
            r = sign.(r) .* max.(0.0, abs.(r) .- alpha * lambda) # backward
        end
    end
    return rs, θs, loss
end
                
Random.seed!(1) # randomness seed # best was 2,4
m0 = 1 # number of spikes ground truth
θ0 = [π, π]'
w0  = [1]

alpha, beta = 0.01, 0.01
lambda =  0.0
r_obs = sign.(w0) .* sqrt.(abs.(w0))
θ_obs = θ0

res = 10 # resolution of the initial measure
m = res^2

r_init = 2*ones(m)
θ_initx = range(π/res, 2π - π/res, length=res)' .* ones(res)
θ_inity = range(π/res, 2π - π/res, length=res) .* ones(res)'
θ_init  = cat(θ_initx[:], θ_inity[:], dims=2)
niter = 10000
alpha, beta = 0.001, 0.0
println("Mirror descent 2D:")
rsA, θsA, lossA = PGD_2D_convex(r_init, θ_init, r_obs, θ_obs, lambda, alpha, beta, niter, mirror=true)
r_init = 0.0*ones(m)
println("Euclidean descent (ISTA) 2D:")
rsB, θsB, lossB = PGD_2D_convex(r_init.^2, θ_init, r_obs.^2, θ_obs, lambda, alpha, beta, niter, mirror=false)
                
                
figure(figsize=[4,3])
loglog(lossA .- lossA[end],label="mirror descent")
loglog(lossB .- lossA[end],label="euclidean descent")
axis([1,1e3,1e-2,40])
xlabel(L"iteration index $k$")
ylabel(L"objective $J(\nu_k)$")
grid("on")
legend()
savefig("vertical_deconv2D.pdf",bbox_inches="tight")