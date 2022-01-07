# ----------------------------------------------------------------------
# chess board experiment
# ----------------------------------------------------------------------
using AtomicOpt
using Printf
using LinearAlgebra
using BSON: @save, @load
import Random: seed!, randperm, shuffle
using LinearMaps
using FFTW

# load data
@load "./examples/ChessBoard/inpainting_data.bson" xs xr ks kr b

# atomic sets
n, m = size(xs)
xs = vec(xs)
xr = vec(xr)
b = vec(b)
As = OneBall(n*m; maxrank = ks+100); As = gauge(As, xs)*As
Ar = NucBall(n, m; maxrank = kr+1); Ar = gauge(Ar, xr)*Ar
A = As × Ar

# noise
function randnsparse(n, k)
    p = randperm(n)[1:k]
    x = zeros(n)
    x[p] = randn(k)
    return x
end
function randop(n::Integer, family=:orth)
    if family == :orth
        M = Matrix(qr(randn(n,n)).Q)
    end
    return M
end
Q1 = randop(n); Q2 = randop(m)
Q = LinearMap(p->vec(Q1*reshape(p,n,m)*Q2'), q->vec(Q1'*reshape(q,n,m)*Q2), n*m, n*m)
u = randnsparse(n*m, floor(Int, 0.02*n*m))
Au = OneBall(n*m; maxrank = floor(Int, 0.02*n*m))
Au = gauge(Au, u)*Au
u = Q*u; Au = Q*Au
b = b + u
A = A × Au

# solve
x, normr = @time level_set(I(n*m), b, A, pr=true, maxIts=100000)
@printf("  Norm of residual: %8.2e\n", normr)
@printf("  Relative error in sparse signal: %8.2e\n", norm(x[1] - xs)/norm(xs))
@printf("  Relative error in lowrank signal: %8.2e\n", norm(x[2] - xr)/norm(xr))

# plot
bp = reshape(b, n, m)
xsp = reshape(x[1], n, m)
xrp = reshape(x[2], n, m)
up = reshape(x[3], n, m)
 
# save
# @save "/Users/zhenanfan/.julia/dev/PolarDemixing/examples/ChessBoard/noisy.bson" xsp xrp bp up
using PyPlot
plots = [bp, abs.(xsp), xrp, bp-up]
fig, axes= subplots(nrows=1, ncols=3)
c = "gray"
for i = 1:3
    subplot(1,3,i)
    imshow(plots[i], cmap=c)
    axis("off")
end
tight_layout(w_pad=1, h_pad=-1, pad=0)
