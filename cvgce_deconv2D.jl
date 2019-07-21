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
    for iter = 1:niter
        rs[:,iter] = r
        θs[:,:,iter] = θ
        Kxx = phi.(θ[:,1] .- θ[:,1]',θ[:,2] .- θ[:,2]')
        Kyy = phi.(θ_obs[:,1] .- θ_obs[:,1]',θ_obs[:,2] .- θ_obs[:,2]')
        Kxy = phi.(θ[:,1] .- θ_obs[:,1]', θ[:,2] .- θ_obs[:,2]')
        a = ss.(r)/m
        b = ss.(r_obs)
        loss[iter] = (a' * Kxx * a + b' * Kyy * b - 2a' * Kxy * b)/2 + lambda * sum(abs.(a))
        #if iter>1 && abs(loss[iter]-loss[iter-1]) <1e-15
        #    return rs, θs, loss[1:iter]
        #end
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



# Parameters for the whole experiment
nf = 2 # number of frequency components
phi(x,y) = real(sum(exp(im*(kx*x + ky*y)) for kx in -nf:nf, ky in -nf:nf))
phi_derx(x,y) = real(sum(im*kx*exp(im*(kx*x + ky*y)) for kx in -nf:nf, ky in -nf:nf))
phi_dery(x,y) = real(sum(im*ky*exp(im*(kx*x + ky*y)) for kx in -nf:nf, ky in -nf:nf))
alpha, beta = 0.01, 0.01
lambdas =  [0.0, 1.0, 4.0]
niter = 3000

# ground truth
m0 = 3
θ_obs = pi .+ [0.0 0.0; 1.7 -0.3; 0.7 1.3]
r_obs = [1.1; 0.8; 1.0]

# CASE 1: largely over-parameterized (m=100)
res = 10 # resolution of the initial measure
m = res^2
r_init = 0.5*ones(m)
θ_initx = range(π/res, 2π - π/res, length=res)' .* ones(res)
θ_inity = range(π/res, 2π - π/res, length=res) .* ones(res)'
θ_init  = cat(θ_initx[:], θ_inity[:], dims=2)

rssA = zeros(m , niter, length(lambdas))
θssA = zeros(m, 2, niter, length(lambdas))
lossesA = Any[]
p = Progress(length(lambdas),1, "Computing (1/3)...")
for i = 1:length(lambdas)
    rssA[:,:,i], θssA[:,:,:,i], loss = PGD_2D(r_init, θ_init, r_obs, θ_obs, lambdas[i], alpha, beta, niter, retraction=0);
    push!(lossesA,loss)
    ProgressMeter.next!(p)
end


# With slight-overparameterization
#θ_obs = pi .+ [0.0 0.0; 1.7 -0.3; 0.7 1.3]
m = 4
θ_init  = pi .+ [0.5 -0.5; 2.1 -0.7; 0.3 1.8; 0.5 1.5]
r_init = 0.5*ones(m)

rssB = zeros(m , niter, length(lambdas))
θssB = zeros(m, 2, niter, length(lambdas))
lossesB = Any[]
p = Progress(length(lambdas),1, "Computing (2/3)...")
for i = 1:length(lambdas)
    rssB[:,:,i], θssB[:,:,:,i], loss = PGD_2D(r_init, θ_init, r_obs, θ_obs, lambdas[i], alpha, beta, niter, retraction=0);
    push!(lossesB,loss)
    ProgressMeter.next!(p)
end


# Without over-parameterization
#θ_obs = pi .+ [0.0 0.0; 1.7 -0.3; 0.7 1.3]
m = 3
θ_init  = pi .+ [0.5 -0.5; 2.1 -0.7; 0.3 1.8]
r_init = 0.5*ones(m)

rssC = zeros(m , niter, length(lambdas))
θssC = zeros(m, 2, niter, length(lambdas))
lossesC = Any[]
p = Progress(length(lambdas),1, "Computing (3/3)...")
for i = 1:length(lambdas)
    rssC[:,:,i], θssC[:,:,:,i], loss = PGD_2D(r_init, θ_init, r_obs, θ_obs, lambdas[i], alpha, beta, niter, retraction=0);
    push!(lossesC,loss)
    ProgressMeter.next!(p)
end

# Compute distances to minimizers
"Compute the Wasserstein distance with cone metric between two matched empirical distributions in the cone"
Wcone_matched(r1,θ1,r2,θ2)  = sqrt(max( sum(r1.^2 .+ r2.^2 .- 2r1 .* r2 .* cos.(min.(sum((θ2 .- θ1).^2, dims=2), π)))/length(r1), 0.0))
WsA = zeros(niter,length(lambdas))
WsB = zeros(niter,length(lambdas))
WsC = zeros(niter,length(lambdas))
for i = 1:niter
    for j = 1:length(lambdas)
        WsA[i,j] = Wcone_matched(rssA[:,i,j],θssA[:,:,i,j],rssA[:,end,j],θssA[:,:,end,j])
        WsB[i,j] = Wcone_matched(rssB[:,i,j], θssB[:,:,i,j], rssB[:,end,j], θssB[:,:,end,j])
        WsC[i,j] = Wcone_matched(rssC[:,i,j], θssC[:,:,i,j], rssC[:,end,j], θssC[:,:,end,j])
    end
end

# PLOT - DISTANCE
figure(figsize=[4,3])
istart=30
imax = div(length(lossesA[1]),2)
for j=1:length(lambdas)
    # renormalize (we compare asymptotic rates)
    semilogy(WsA[istart:imax,j]./WsA[istart,j],"C$(j-1)")
    semilogy(WsB[istart:imax,j]./WsB[istart,j],"--C$(j-1)")
    semilogy(WsC[istart+j:imax,j]./WsC[istart,j],":C$(j-1)")
end
axis([-30,1500,1e-5,30])
xlabel(L"Iteration index $k$")
ylabel("Distance to minimizer")
custom_lines = [matplotlib.lines.Line2D([0], [0], color="C0", lw=1),
                matplotlib.lines.Line2D([0], [0], color="C1", lw=1),
                matplotlib.lines.Line2D([0], [0], color="C2", lw=1),
                matplotlib.lines.Line2D([0], [0], color="k", lw=1),
                matplotlib.lines.Line2D([0], [0], color="k", linestyle="--", lw=1),
                matplotlib.lines.Line2D([0], [0], color="k", linestyle=":", lw=1)]

legend(custom_lines, [L"\lambda=0", L"\lambda=1", L"\lambda=4", L"m=100", L"m=4", L"m=3"],ncol=2,loc=1)
grid("on")
savefig("cvgce_deconv2D_distance.pdf",bbox_inches="tight")


# PLOT - LOSS
figure(figsize=[4,3])
istar = 30
imax = div(length(lossesA[1]),2)
for j=1:length(lambdas)
    # renormalize (we compare asymptotic rates)
    semilogy((istar:imax),(lossesA[j][istar:imax] .- lossesA[j][end])./ (lossesA[j][istar] .- lossesA[j][end]),"C$(j-1)")
    semilogy((istar:imax),(lossesB[j][istar:imax].- lossesB[j][end]) ./ (lossesB[j][istar] .- lossesB[j][end]),"--C$(j-1)")
    semilogy((istar:imax),(lossesC[j][istar:imax].- lossesC[j][end]) ./ (lossesC[j][istar] .- lossesC[j][end]),":C$(j-1)")
end
axis([0,imax,1e-7,1e2])
xlabel(L"Iteration index $k$")
ylabel("Normalized optimality gap")
custom_lines = [matplotlib.lines.Line2D([0], [0], color="C0", lw=1),
                matplotlib.lines.Line2D([0], [0], color="C1", lw=1),
                matplotlib.lines.Line2D([0], [0], color="C2", lw=1),
                matplotlib.lines.Line2D([0], [0], color="k", lw=1),
                matplotlib.lines.Line2D([0], [0], color="k", linestyle="--", lw=1),
                matplotlib.lines.Line2D([0], [0], color="k", linestyle=":", lw=1)]

legend(custom_lines, [L"\lambda=0", L"\lambda=1", L"\lambda=4", L"m=100", L"m=4", L"m=3"],ncol=2)
grid("on")
savefig("cvge_deconv2D_loss.pdf",bbox_inches="tight")