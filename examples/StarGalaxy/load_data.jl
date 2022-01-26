using LinearAlgebra
using Images
using FFTW

# load data
xs = load("./examples/StarGalaxy/star.png")
xd = load("./examples/StarGalaxy/galaxy.png")

# size
m, n = size(xs)

# preprocessing on xs
xs = Gray.(xs); xs = convert(Array{Float64}, xs)
γ = xs[3,1]; xs = xs .- γ # normalize
idx = findall(x->abs(x)<=7e-2, xs); xs[idx] .= 0
ks = count(x->x!=0, xs)
xs = vec(xs)

# preprocessing on xd
xd = Gray.(xd); xd = convert(Array{Float64}, xd);
zd = dct(xd); idx = findall(x->abs(x)<=0.02, zd); zd[idx] .= 0
kd = count(x->x!=0, zd)
xd = idct(zd)
xd = vec(xd)

# observation
b = xs + xd

# write to file
f = open("/home/zhenan/Github/AtomicOpt.jl/examples/StarGalaxy/StarGalaxyData.txt", "w")
l = length(b)
@printf(f, "xs = ")
for i in 1:l
    @printf(f, "%.3f ", xs[i])
end
println(f, "\n")
@printf(f, "xd = ")
for i in 1:l
    @printf(f, "%.3f ", xd[i])
end
println(f, "\n")
@printf(f, "b = ")
for i in 1:l
    @printf(f, "%.3f ", b[i])
end
println(f, "\n")
@printf(f, "ks = %d\n", ks)
@printf(f, "kd = %d\n", kd)
@printf(f, "m = %d\n", m)
@printf(f, "n = %d\n", n)
close(f)

