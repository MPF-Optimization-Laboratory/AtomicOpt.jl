using LinearAlgebra
using Images
using Printf

# load data
xs = load("./examples/ChessBoard/inpainting_sparse.png")
xr = load("./examples/ChessBoard/inpainting_low_rank.png")

# size
m, n = size(xs)

# preprocessing on xs
xs = Gray.(xs); xs = convert(Array{Float64}, xs)
γ = xs[2,2]; xs = xs .- γ # normalize
idx = findall(x->abs(x)<=5e-2, xs); xs[idx] .= 0
idx = findall(x->x!=0, xs); xs[idx] .= maximum(xs)
ks = count(x->x!=0, xs)
xs = 3*vec(xs)

# preprocessing on xr
xr = Gray.(xr); xr = convert(Array{Float64}, xr)
F = svd(xr); kr = 2
xr = F.U[:, 1:kr] * Diagonal(F.S[1:kr]) * F.Vt[1:kr, :]
xr = 0.5*vec(xr)

# observation
b = xs + xr


# write to file
f = open("/home/zhenan/Github/AtomicOpt.jl/examples/ChessBoard/ChessBoardData.txt", "w")
l = length(b)
@printf(f, "xs = ")
for i in 1:l
    @printf(f, "%.3f ", xs[i])
end
println(f, "\n")
@printf(f, "xr = ")
for i in 1:l
    @printf(f, "%.3f ", xr[i])
end
println(f, "\n")
@printf(f, "b = ")
for i in 1:l
    @printf(f, "%.3f ", b[i])
end
println(f, "\n")
@printf(f, "ks = %d\n", ks)
@printf(f, "kr = %d\n", kr)
@printf(f, "m = %d\n", m)
@printf(f, "n = %d\n", n)
close(f)

