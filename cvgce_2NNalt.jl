using PyPlot, ProgressMeter
using Random, LinearAlgebra


function populationSGDfor2NN_alt(w_init, S, w_tea, S_tea, lambda, stepsize, niter, batchsize)# S for signs
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
        
        w = w - stepsize * gradw#/(1+sqrt(iter))
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

stepsize = 0.0002
niter = 500000
batchsize = 500
ms = [5,20]
#lambdas = [0.0, 0.1, 0.2] # OLD: too simple (1 spike)
lambdas = [0.0, 0.05, 0.1]
lossregs = Array{Any, 2}(undef, length(lambdas) ,length(ms))

# hyper-parameters
p = Progress(length(lambdas)*length(ms),1, "Computing...")
for k_l=1:length(lambdas)
    for k_m = 1:length(ms)
        m = ms[k_m]
        w_init = randn(m,d)
        S = ones(m)
        ws, loss, lossregs[k_l,k_m] = populationSGDfor2NN_alt(w_init, S, w_tea, S_tea, lambdas[k_l], stepsize, niter, batchsize)
        ProgressMeter.next!(p)
    end
end

figure(figsize=[4,3])
imax = 300000#250000
for i in 1:length(lambdas)
    ca = lossregs[i,1][1:100:imax]./lossregs[i,1][1]  .- lossregs[i,1][end]./lossregs[i,1][1] #.+ 1e-4
    cb = lossregs[i,end][1:100:imax]./lossregs[i,end][1]  .- lossregs[i,end][end]./lossregs[i,end][1] #.+ 1e-4
    res = 20 # smoothing window
    cal = [sum(ca[i:i+res])/res for i=1:length(ca)-res]
    cbl = [sum(cb[i:i+res])/res for i=1:length(cb)-res]
    semilogy(cal,"C$(i-1)",linewidth=1)
    semilogy(cbl .*1.001,"--C$(i-1)",linewidth=1)
end
axis([0, 3000, 0.002, 4])
xticks([0, 1000, 2000, 3000],[L"0", L"10^5",L"2.10^5",L"3.10^5"])
custom_lines = [matplotlib.lines.Line2D([0], [0], color="C0", lw=1),
                matplotlib.lines.Line2D([0], [0], color="C1", lw=1),
                matplotlib.lines.Line2D([0], [0], color="C2", lw=1),
                matplotlib.lines.Line2D([0], [0], color="k", lw=1),
                matplotlib.lines.Line2D([0], [0], color="k", linestyle="--", lw=1)]
legend(custom_lines, [L"\lambda=0", L"\lambda=0.05", L"\lambda=0.1", L"m=5", L"m=20"],ncol=2)
xlabel(L"iteration index $k$")
ylabel("Normalized optimality gap")
grid("on")
savefig("cvgce_2NNalt_loss.pdf",bbox_inches="tight")