# ----------------------------------------------------------------------
# Test level-set algorithm on demixing sparse plus low rank signals
# ----------------------------------------------------------------------
using AtomicOpt
using Printf
using LinearAlgebra
import Random: seed!, randperm, shuffle
# using ProfileView

function randnsparse(n, k)
    p = randperm(n)[1:k]
    x = zeros(n)
    x[p] = randn(k)
    return x
end

function randnlowrank(n, m, k)
    u = randn(n, k)
    v = randn(m, k)
    x = vec(u*v')
    return x
end

# setting
n = 16; m = 16; ks = 32; kr = 1;
xs = randnsparse(n*m, ks); @show(rank(reshape(xs, n, m)))
xr = randnlowrank(n, m, kr)
b = xs + xr
As = OneBall(n*m; maxrank = ks); As = gauge(As, xs)*As
Ar = NucBall(n, m; maxrank = kr); Ar = gauge(Ar, xr)*Ar
A = As × Ar

# solve
x, normr = @time level_set(I(n*m), b, A, maxIts=30000)
@printf("  Relative error in sparse signal: %8.2e\n", norm(x[1] - xs)/norm(xs))
@printf("  Relative error in lowrank signal: %8.2e\n", norm(x[2] - xr)/norm(xr))
