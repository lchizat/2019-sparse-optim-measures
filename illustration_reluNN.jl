using PyPlot, ProgressMeter
using Random, LinearAlgebra


"""
Gradient descent to train a 2-layers ReLU neural net for the square loss and with weight decay.
f(x,w=(a,b)) = (1/m) sum_{i=1}^m b_i (x * a_i)_+
F(w) = MSE(f(w)) + (lambda/m) * sum_i b_i * || a_i||
"""
function GDfor2NN(X_train, Y_train, W_init, lambda, stepsize, niter) 
    (n,d) = size(X_train)
    m     = size(W_init, 1)
    W     = copy(W_init)
    Ws    = zeros(m, d+1, niter)# store optimization path
    loss_train = zeros(niter)
    for iter = 1:niter
        Ws[:,:,iter] = W
        # output of the neural net
        temp    =  max.( W[:,1:end-1] * X_train', 0.0) # output hidden layer (size m × n)
        output  =  (1/m) * sum( W[:,end] .* temp , dims=1) # output network (size 1 × n)
        # compute gradient
        gradR   = (output .- Y_train)'/n  # size n
        grad_w1 = (W[:,end] .* float.(temp .> 0) * ( X_train .* gradR )) + 2 * lambda * W[:,1:end-1]  # (size m × d) 
        grad_w2 = temp * gradR  + 2 * lambda * W[:,end] # size m
        grad = cat(grad_w1, grad_w2, dims=2) # size (m × d+1)   
        # store train loss
        loss_train[iter] = (1/2) * sum((output - Y_train).^2)/n + (lambda/m) * sum(W.^2)
        # gradient descent
        W = W - stepsize * grad
    end
    Ws, loss_train
end

Random.seed!(1)
# generate the data
d = 2 # dimension of input

# random teacher 2-NN
m0 = 3 # nb of neurons teacher
w1 = randn(m0,d)
w1 = w1 ./ sqrt.(sum(w1.^2, dims=2))
w2  = sign.(randn(m0))
f(X) = (1/m0) * sum( w2 .* max.( w1 * X', 0.0), dims=1)

# data sets
n_train  = 15 # size train set (15)
X_train = randn(n_train, d)
X_train = X_train  ./ sqrt.(sum(X_train.^2, dims=2))
Y_train = f(X_train)

# initialize and train
m = 20 # nb of neurons student
niter = 10^4
stepsize = 1
lambda = 0.002

# initialization
W_init = randn(m, 3)
W_init = 2*W_init  ./ sqrt.(sum(W_init.^2, dims=2))
#W_init = cat(W_init, rand(m),dims=2)

# choose scale for the initialization
W_init = W_init

Ws, loss_train= GDfor2NN(X_train, Y_train, W_init, lambda, stepsize, niter);



# things to plot
iters = Int.(floor.(exp.(range(0, stop = log(niter), length = 100)))) 
#mid=div(m,2)
finalsign = sign.(Ws[:,end,end])
pxs = Ws[finalsign.>0,1,iters] .* Ws[finalsign.>0,end,iters]
pys = Ws[finalsign.>0,2,iters] .* Ws[finalsign.>0,end,iters]
pxsm = Ws[finalsign.<0,1,iters] .* abs.(Ws[finalsign.<0,end,iters])
pysm = Ws[finalsign.<0,2,iters] .* abs.(Ws[finalsign.<0,end,iters])
px0 = w1[:,1] #ground truth
py0 = w1[:,2] #ground truth


figure(figsize=[4,3])
r = 2
plot(r*cos.(0.0:0.01:2π),r*sin.(0.0:0.01:2π),":",color="k")
rr = 3
plot(rr*cos.(0.0:0.01:2π),rr*sin.(0.0:0.01:2π),":",color="w")

plot(pxs',pys',linewidth=0.7,"k");
plot(pxsm',pysm',linewidth=0.7,"k");
scatter(pxs[:,end],pys[:,end],30,color="C3")
scatter(pxsm[:,end],pysm[:,end],30,color="C0")

bx= max(max(maximum(abs.(pxs)), maximum(abs.(pys)))*1.1,1.1)
bxm= max(max(maximum(abs.(pxsm)), maximum(abs.(pysm)))*1.1,1.1)
bx = max(bx,bxm)
axis("square")
#legend()
axis([-bx,bx,-bx,bx]);
axis("off")
text(0.8*r,0.8*r,L"\Theta")
savefig("illustration_reluNN.pdf",bbox_inches="tight")