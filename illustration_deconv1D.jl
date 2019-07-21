using PyPlot, ProgressMeter
using Random, LinearAlgebra

nf = 4 # number of frequency components
phi(x) = real(sum(exp.(im*k*x) for k in -nf:nf))
phi_der(x) = real(sum(im*k*exp.(im*k*x) for k in -nf:nf))

function PGD_1D(r_init, θ_init, r_obs, θ_obs, lambda, alpha, beta, niter; retraction =0)
    ss(r) = abs(r)*r # signed square function
    f_obs(x) = sum(ss.(r_obs) .* phi.(x .- θ_obs)) # observed signal 
    m = length(r_init)
    rs = zeros(m,niter)
    θs = zeros(m,niter)
    gradr, gradθ = zeros(m), zeros(m)
    r, θ = r_init, θ_init
    loss = zeros(niter)
    for iter = 1:niter
        rs[:,iter] = r
        θs[:,iter] = θ
        
        Kxx = phi.(θ .- θ')
        Kyy = phi.(θ_obs .- θ_obs')
        Kxy = phi.(θ .- θ_obs')
        a = ss.(r)/m
        b = ss.(r_obs)
        loss[iter] = (a' * Kxx * a + b' * Kyy * b - 2a' * Kxy * b)/2 + lambda * sum(abs.(a))
        
        for i = 1:m  # gradient computation
            gradr[i] = sign.(r[i]) * (sum(ss.(r) .* phi.(θ[i] .- θ))/m .- sum(ss.(r_obs) .* phi.(θ[i] .- θ_obs))) + lambda
            gradθ[i] = sign.(r[i]) * (sum(ss.(r) .* phi_der.(θ[i] .- θ))/m .- sum(ss.(r_obs) .* phi_der(θ[i].-θ_obs)))       
        end
        if retraction == 0
            r = r .* exp.(- 2 * alpha * gradr) # mirror retraction
        else
            r = r .* (1 .- 2 * alpha * gradr) # canonical retraction
        end
        θ = θ .- beta * gradθ
    end
    return rs, θs, loss
end
   

Random.seed!(4) 
m0 = 5 # number of spikes ground truth
min_separation = 0.2
@assert m0 * min_separation < π/2 # they can't be all far away

# generate random spikes on the domain [0,2π]
θ0 = sort(rand(m0)) * 2π
while min(minimum(abs.(θ0[2:end] - θ0[1:end-1])), abs((θ0[end]-2π)-θ0[1])) < min_separation # min separation
    global θ0 = sort(rand(m0)) * 2π
end
w0  = normalize(sign.(randn(m0))/2 + (rand(m0) .- 1/2)/2 , 1)
    
alpha, beta = 0.05, 0.005
lambda =  0.3
r_obs = sign.(w0) .* sqrt.(abs.(w0))
θ_obs = θ0
m = 30
r_init = 0.5*ones(m)
r_init[isodd.(1:m)] *= -1.0
θ_init = range(0,2π*(1-1/m),length=m)
niter = 1000
rs, θs, loss = PGD_1D(r_init, θ_init, r_obs, θ_obs, lambda, alpha, beta, niter,retraction = 0)

figure(figsize=[4,3])
xs = range(0, 2π, length=200)
plot(xs, sum(w0'.*phi.(xs.-θ0'),dims=2)/2,label="observed",color="C1",":")
plot(θs',rs',"k",linewidth=0.8)
plot((θs .+ 2π)',rs',"k",linewidth=0.8)
plot((θs .- 2π)',rs',"k",linewidth=0.8)
II = rs[:,end].>=0
I = rs[:,end].<0
plot(θs[II,end],rs[II,end],".C3",markersize=8,label="limit of the flow")
plot(θs[II,end] .+ 2π,rs[II,end],".C3",markersize=8)
plot(θs[II,end] .- 2π,rs[II,end],".C3",markersize=8)
plot(θs[I,end],rs[I,end],".C0",markersize=8)
plot(θs[I,end].+ 2π,rs[I,end],".C0",markersize=8)
plot(θs[I,end].- 2π,rs[I,end],".C0",markersize=8)
axis([0,2π,-2.5,2.5])
hlines(0,0,2π,"k")
xlabel(L"\Theta")
xticks([])
yticks([0])
savefig("illustration_deconv1D.pdf",bbox_inches="tight")