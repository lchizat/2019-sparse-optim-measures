using PyPlot, ProgressMeter
using Random, LinearAlgebra


nf = 2 # number of frequency components
phi(x,y) = real(sum(exp(im*(kx*x + ky*y)) for kx in -nf:nf, ky in -nf:nf))
phi_derx(x,y) = real(sum(im*kx*exp(im*(kx*x + ky*y)) for kx in -nf:nf, ky in -nf:nf))
phi_dery(x,y) = real(sum(im*ky*exp(im*(kx*x + ky*y)) for kx in -nf:nf, ky in -nf:nf))


function PGD_2D(r_init, θ_init, r_obs, θ_obs, lambda, alpha, beta, niter; retraction=0)
    ss(r) = abs(r)*r # signed square function
    m0 = length(r_obs)
    m = length(r_init)
    rs = zeros(m, niter)
    θs = zeros(m, 2, niter)
    gradr, gradθ = zeros(m), zeros(m, 2)
    r, θ = r_init, θ_init
    loss = zeros(niter)
    @showprogress 1 "Computing for resolution $(length(r_init))..." for iter = 1:niter
        rs[:,iter] = r
        θs[:,:,iter] = θ
        Kxx = phi.(θ[:,1] .- θ[:,1]',θ[:,2] .- θ[:,2]')
        Kyy = phi.(θ_obs[:,1] .- θ_obs[:,1]',θ_obs[:,2] .- θ_obs[:,2]')
        Kxy = phi.(θ[:,1] .- θ_obs[:,1]', θ[:,2] .- θ_obs[:,2]')
        a = ss.(r)/m
        b = ss.(r_obs)
        loss[iter] = (a' * Kxx * a + b' * Kyy * b - 2a' * Kxy * b)/2 + lambda * sum(abs.(a))
        for i = 1:m  # gradient computation
            # note that to simplify, we use the properties of the Dirichlet kernel
            # in real applications, one would need to compute integrals over the torus
            gradr[i] = (sign.(r[i]) * (sum(ss.(r) .* phi.(θ[i,1] .- θ[:,1], θ[i,2] .- θ[:,2]))/m 
                        .- sum(ss.(r_obs) .* phi.(θ[i,1] .- θ_obs[:,1], θ[i,2] .- θ_obs[:,2])))
                        + lambda)
            gradθx = sign.(r[i]) * (sum(ss.(r) .* phi_derx.(θ[i,1] .- θ[:,1], θ[i,2] .- θ[:,2]))/m
                    .-  sum(ss.(r_obs) .* phi_derx.(θ[i,1] .- θ_obs[:,1], θ[i,2].-θ_obs[:,2])))
            gradθy = sign.(r[i]) * (sum(ss.(r) .* phi_dery.(θ[i,1] .- θ[:,1], θ[i,2] .- θ[:,2]))/m 
                    .- sum(ss.(r_obs) .* phi_dery.(θ[i,1].-θ_obs[:,1], θ[i,2].-θ_obs[:,2])))
            gradθ[i,:] = [gradθx,gradθy]
        end
        if retraction == 0
            r = r .* exp.( -2 * alpha * gradr) # mirror retraction
        else
            r = r .* (1 .- 2 * alpha * gradr) # canonical retraction
        end
        θ = θ .- beta * gradθ
        #beta = beta0 * min(20, 1 + iter*10/1000) # heuristic to increase beta
    end
    return rs, θs, loss
end
    
m0 = 3
θ_obs = pi .+ [0.0 0.0; 1.7 -0.3; 0.7 1.3]
r_obs = [1.1; 0.8; 1.0]
xs = range(0,2pi,length=50)
ys = range(0,2pi,length=50)
f_obs(x,y) = sum(r_obs[i] * phi.(xs' .- θ_obs[i,1],ys .- θ_obs[i,2]) for i=1:m0)

alpha, beta = 0.01, 0.01 #(beta in 0.00001, 0.0001)
lambda =  1.0
niter = 3*10^3
    
# resolution 5x5
res = 5 # resolution of the initial measure
m = res^2
r_init = 0.5*ones(m)
#r_init[isodd.(1:m)] *= -1.0
θ_initx = range(π/res, 2π - π/res, length=res)' .* ones(res)
θ_inity = range(π/res, 2π - π/res, length=res) .* ones(res)'
θ_init  = cat(θ_initx[:], θ_inity[:], dims=2)
rs, θs, loss = PGD_2D(r_init, θ_init, r_obs, θ_obs, lambda, alpha, beta, niter, retraction=0)
    
figure(figsize=[3,3])
xs = range(0,2pi,length=200)
ys = range(0,2pi,length=200)
pcolor(xs,ys,sum(r_obs[i] * phi.(xs' .- θ_obs[i,1],ys .- θ_obs[i,2]) for i=1:m0)); 
axis("off");
scatter(θ_obs[:,1], θ_obs[:,2], 30*r_obs, "k")
plot(permutedims(θs[:,1,:],[2,1]),permutedims(θs[:,2,:],[2,1]),"k",linewidth=1)
plot(permutedims(θs[:,1,:],[2,1]),permutedims(θs[:,2,:].+2π,[2,1]),"k",linewidth=1)
plot(permutedims(θs[:,1,:].+2π,[2,1]),permutedims(θs[:,2,:],[2,1]),"k",linewidth=1)
plot(permutedims(θs[:,1,:],[2,1]),permutedims(θs[:,2,:].-2π,[2,1]),"k",linewidth=1)
plot(permutedims(θs[:,1,:].-2π,[2,1]),permutedims(θs[:,2,:],[2,1]),"k",linewidth=1)
scatter(θs[:,1,end],θs[:,2,end], 20*rs[:,end],"C3")
scatter(θs[:,1,end],θs[:,2,end].+2π, 20*rs[:,end],"C3")
scatter(θs[:,1,end].+2π,θs[:,2,end], 20*rs[:,end],"C3")
scatter(θs[:,1,end],θs[:,2,end].-2π, 20*rs[:,end],"C3")
scatter(θs[:,1,end].-2π,θs[:,2,end], 20*rs[:,end],"C3")
scatter(θ_obs[:,1], θ_obs[:,2], 30*r_obs, "w")
axis([0,2π,0,2π]);
savefig("illustration_deconv2D_25.png",bbox_inches="tight",dpi=300)

# resolution 10x10
res = 10 # resolution of the initial measure
m = res^2
r_init = 0.5*ones(m)
#r_init[isodd.(1:m)] *= -1.0
θ_initx = range(π/res, 2π - π/res, length=res)' .* ones(res)
θ_inity = range(π/res, 2π - π/res, length=res) .* ones(res)'
θ_init  = cat(θ_initx[:], θ_inity[:], dims=2)
rs, θs, loss = PGD_2D(r_init, θ_init, r_obs, θ_obs, lambda, alpha, beta, niter, retraction=0)
    
figure(figsize=[3,3])
xs = range(0,2pi,length=200)
ys = range(0,2pi,length=200)
pcolor(xs,ys,sum(r_obs[i] * phi.(xs' .- θ_obs[i,1],ys .- θ_obs[i,2]) for i=1:m0)); 
axis("off");
scatter(θ_obs[:,1], θ_obs[:,2], 30*r_obs, "k")
plot(permutedims(θs[:,1,:],[2,1]),permutedims(θs[:,2,:],[2,1]),"k",linewidth=1)
plot(permutedims(θs[:,1,:],[2,1]),permutedims(θs[:,2,:].+2π,[2,1]),"k",linewidth=1)
plot(permutedims(θs[:,1,:].+2π,[2,1]),permutedims(θs[:,2,:],[2,1]),"k",linewidth=1)
plot(permutedims(θs[:,1,:],[2,1]),permutedims(θs[:,2,:].-2π,[2,1]),"k",linewidth=1)
plot(permutedims(θs[:,1,:].-2π,[2,1]),permutedims(θs[:,2,:],[2,1]),"k",linewidth=1)
scatter(θs[:,1,end],θs[:,2,end], 20*rs[:,end],"C3")
scatter(θs[:,1,end],θs[:,2,end].+2π, 20*rs[:,end],"C3")
scatter(θs[:,1,end].+2π,θs[:,2,end], 20*rs[:,end],"C3")
scatter(θs[:,1,end],θs[:,2,end].-2π, 20*rs[:,end],"C3")
scatter(θs[:,1,end].-2π,θs[:,2,end], 20*rs[:,end],"C3")
scatter(θ_obs[:,1], θ_obs[:,2], 30*r_obs, "w")
axis([0,2π,0,2π]);
savefig("illustration_deconv2D_100.png",bbox_inches="tight",dpi=300)

    
