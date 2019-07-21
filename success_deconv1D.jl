using PyPlot, ProgressMeter
using Random, LinearAlgebra

nf = 8 # number of frequency components
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


m0 = 5
min_separation = 0.0
lambda =  0.3

ms = 4:10
alpha = 0.002
betas = alpha ./ (2.0 .^(0:4))
niter = 1000
nexp = 10
rr = Array{Any, 3}(undef, length(betas) ,length(ms), nexp)
tt = Array{Any, 3}(undef, length(betas) ,length(ms), nexp)
losses = Array{Any, 3}(undef, length(betas) ,length(ms), nexp)


Random.seed!(2) # randomness seed # best was 2,4
p = Progress(nexp*length(ms)*length(betas))
for k_exp=1:nexp
    # generate random spikes on the domain [0,2π]
    θ0 = sort(rand(m0)) * 2π
    while min(minimum(abs.(θ0[2:end] - θ0[1:end-1])), abs((θ0[end]-2π)-θ0[1])) < min_separation # min separation
        θ0 = sort(rand(m0)) * 2π
    end
    w0  = rand(m0) #normalize(sign.(randn(m0))/2 + (rand(m0) .- 1/2)/2 , 1)
    r_obs = sqrt.(abs.(w0))# sign.(w0) .* sqrt.(abs.(w0))
    θ_obs = θ0
    for k_m = 1:length(ms)
        for k_b=1:length(betas)
            m = ms[k_m]
            beta = betas[k_b]
            r_init = 0.5*ones(m)
            #r_init[isodd.(1:m)] *= -1.0
            θ_init = range(0,2π*(1-1/m),length=m)
            rs, θs, loss = PGD_1D(r_init, θ_init, r_obs, θ_obs, lambda, alpha, beta, niter,retraction = 0)
            rr[k_b,k_m,k_exp] = rs[:,end]
            tt[k_b,k_m,k_exp] = θs[:,end]
            losses[k_b,k_m,k_exp] = loss[:]
            ProgressMeter.next!(p)
        end
    end
end

Tlosses = zeros(length(betas) ,length(ms), nexp)
for i = 1:length(betas)
    for j = 1:length(ms)
        for k = 1:nexp
            Tlosses[i,j,k] = losses[i,j,k][end]
        end
    end
end
for k=1:nexp
    Tlosses[:,:,k] = Tlosses[:,:,k] .- minimum(Tlosses[:,:,k])
end
loss_ave = sum(Tlosses, dims=3)/nexp

figure(figsize=[4,2])
pcolor(loss_ave[end:-1:1,:]./maximum(loss_ave),cmap="gray_r")
xticks((0:7).+0.5,["4", "5", "6", "7", "8", "9", "10"])
xlabel(L"m")
yticks([0.5, 1.5, 2.5, 3.5, 4.5], ["1/16", "1/8", "1/4", "1/2", "1"])
ylabel(L"\beta/\alpha")
colorbar(ticks=[0,1])
savefig("success_deconv1D.pdf",bbox_inches="tight")
