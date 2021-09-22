using AtomicOpt
using LinearAlgebra
using PyPlot
using JLD
using LinearMaps

# load data
b = load("./examples/MultiscaleLowRank/MultiscaleLowRankRandom.jld", "b")
A = load("./examples/MultiscaleLowRank/MultiscaleLowRankRandom.jld", "A")
x, feas= level_set(I(length(b)), b, A, tol = 1e-3, maxIts=5000)

# plot
fig, axes= subplots(nrows=1, ncols=5,figsize=(8,2))
subplot(1, 5, 1)
imshow(reshape(b, 64, 64))
title("observation")
axis("off")
for i = 1:4
    subplot(1, 5, i+1)
    imshow(reshape(x[i], 64, 64))
    axis("off")
    title(string(2^(2*i - 2)) * "×" * string(2^(2*i - 2)))
end
subplots_adjust(wspace=0.05, hspace=0)
 