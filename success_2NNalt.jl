using PyPlot, ProgressMeter
using Random, LinearAlgebra


function populationSGDfor2NN_alt_beta(w_init, S, w_tea, S_tea, lambda, stepsize, niter, batchsize, beta)# S for signs
    ss(u) = u*abs(u)
    m, d  = size(w_init)
    m0 = size(w_tea,1)
    w  = copy(w_init)
    ws = zeros(m,d,niter) # storing neurons
    loss = zeros(niter)
    lossreg = zeros(niter)

    # gradient flow
    for iter = 1:niter
        ws[:,:,iter] = w
        # random data points
        X = randn(batchsize,d)
        X = X ./ sqrt.(sum(X.^2, dims=2))
        Y0 = sum( S_tea .* max.(ss.(w_tea)*X', 0.0), dims=1)  #ground truth output

        # prediction and gradient computation
        temp = max.( ss.(w) * X', 0.0)
        Y = sum( S .* temp, dims=1)/m
        loss[iter] = (1/2)*sum( ( Y - Y0).^2 )/batchsize
        lossreg[iter] = (1/2)*sum( ( Y - Y0).^2 )/batchsize + (lambda/m) * sum(w.^2)
        gradR = ( Y - Y0 )'/batchsize # column of size batchsize
        gradw = ((S .* float.(temp .> 0.0)) * ( X .* gradR )).* (2*abs.(w) ) + 2 * lambda * w
        r_component = (sum(gradw .* w, dims=2) ./ sum(w.^2,dims=2)) .* w
        w = w - stepsize * (r_component + beta * (gradw .- r_component))
    end
    ws, loss, lossreg
end


Random.seed!(1);
d = 20
# random ground truth 
m0 = 5 # number of neurons of teacher
w_tea = randn(m0,d)
w_tea = w_tea ./ sqrt.(sum(w_tea.^2,dims=2))
S_tea = ones(m0)

stepsize = 0.004
niter = 100000
batchsize = 500
ms = [1 3 5 7]
lambda = 0.0005
betas = [1.0 2.0 4.0]
nexp=5
#betas = [1]
lossregs = Array{Any, 3}(undef, length(betas) ,length(ms), nexp)

# hyper-parameters
p = Progress(nexp*length(ms)*length(betas))
for k_exp = 1:nexp
    w_tea = randn(m0,d)
    w_tea = w_tea ./ sqrt.(sum(w_tea.^2,dims=2))
    S_tea = ones(m0)
    for k_b=1:length(betas)
        for k_m = 1:length(ms)
            m = ms[k_m]
            w_init = randn(m,d)
            S = ones(m)
            ws, loss, lossregs[k_b,k_m,k_exp] = populationSGDfor2NN_alt_beta(w_init, S, w_tea, S_tea, lambda, stepsize, niter, batchsize, betas[k_b])
            ProgressMeter.next!(p)
        end
    end
end

Tlosses = zeros(length(betas) ,length(ms), nexp)
for i = 1:length(betas)
    for j = 1:length(ms)
        for k = 1:nexp
            Tlosses[i,j,k] = sum(lossregs[i,j,k][end-100:end])/100
        end
    end
end
for k=1:nexp
    Tlosses[:,:,k] = Tlosses[:,:,k] .- minimum(Tlosses[:,:,k])
end
loss_ave = (sum(Tlosses, dims=3)/nexp)[:,:]


figure(figsize=[4,2])
pcolor(loss_ave./maximum(loss_ave), cmap="gray_r")
#pcolor( betas[:],ms[:], 20000*loss_ave', cmap="gray_r")
#pcolor(ms, betas, loss_ave',cmap="gray_r")
xticks([0.5, 1.5, 2.5, 3.5],["1", "3", "5", "7"])
xlabel(L"m")
yticks([0.5; 1.5 ;2.5], ["1", "2", "4"])
ylabel(L"\beta/\alpha")
cbar = colorbar(ticks=[0.01,1])
cbar.ax.set_yticklabels(["0","1"])
savefig("success_2NNalt.pdf",bbox_inches="tight")