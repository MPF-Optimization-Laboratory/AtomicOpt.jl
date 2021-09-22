using LinearAlgebra
using Images
using PyPlot
using FFTW

# load data
xs = load("./examples/StarGalaxy/star.png")
xd = load("./examples/StarGalaxy/galaxy.png")

# preprocessing on xs
xs = Gray.(xs); xs = convert(Array{Float64}, xs)
γ = xs[3,1]; xs = xs .- γ # normalize
idx = findall(x->abs(x)<=7e-2, xs); xs[idx] .= 0
@show ks = count(x->x!=0, xs)

# preprocessing on xd
xd = Gray.(xd); xd = convert(Array{Float64}, xd);
zd = dct(xd); idx = findall(x->abs(x)<=0.02, zd); zd[idx] .= 0
@show kd = count(x->x!=0, zd)
xd = idct(zd)

# observation
b = xs + xd

# plot
c = "gray"
subplot(131)
imshow(b, cmap=c)
subplot(132)
imshow(xs, cmap=c)
subplot(133)
imshow(xd, cmap=c)

# save
# using BSON: @save, @load
# @save "./examples/StarGalaxy/data.bson" xs xd ks kd b
