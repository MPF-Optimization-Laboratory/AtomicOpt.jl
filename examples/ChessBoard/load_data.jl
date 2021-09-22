using LinearAlgebra
using Images
using PyPlot

# load data
xs = load("./examples/ChessBoard/inpainting_sparse.png")
xr = load("./examples/ChessBoard/inpainting_low_rank.png")

# preprocessing on xs
xs = Gray.(xs); xs = convert(Array{Float64}, xs)
γ = xs[2,2]; xs = xs .- γ # normalize
idx = findall(x->abs(x)<=5e-2, xs); xs[idx] .= 0
ks = count(x->x!=0, xs)

# preprocessing on xr
xr = Gray.(xr); xr = convert(Array{Float64}, xr)
F = svd(xr); kr = 2
xr = F.U[:, 1:kr] * Diagonal(F.S[1:kr]) * F.Vt[1:kr, :]

# observation
b = xs + xr

# plot
c = "bone"
subplot(221)
imshow(b, cmap=c)
subplot(222)
imshow(xs, cmap=c)
subplot(223)
imshow(xr, cmap=c)

# save
using BSON: @save, @load
@save "./examples/ChessBoard/inpainting_data.bson" xs xr ks kr b
