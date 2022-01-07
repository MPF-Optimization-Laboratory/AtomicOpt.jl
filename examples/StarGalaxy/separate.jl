using AtomicOpt
using LinearAlgebra
using PyPlot
using BSON: @save, @load
using LinearMaps
using FFTW

# load data
@load "./examples/StarGalaxy/data.bson" xs xd ks kd b

# atomic sets
n, m = size(xs)
xs = vec(xs)
xd = vec(xd)
b = vec(b)
τs = gauge(OneBall(n*m), xs)
As = τs*OneBall(n*m; maxrank = ks)
τd = gauge(OneBall(n*m), dct(xd));
Q = LinearMap(p->idct(p), q->dct(q), n*m, n*m)
Ad = τd*Q*OneBall(n*m; maxrank = kd)
A = As × Ad

x, feas= level_set(I(length(b)), b, A, tol = 1e-3, maxIts=5000)

# @save "./examples/StarGalaxy/result.bson" b x 

# plot
c = "gray"
fig, axes= subplots(nrows=1, ncols=3)
subplot(1, 3, 1)
imshow(reshape(b, n, m), cmap=c)
title("observation")
axis("off")
subplot(1, 3, 2)
imshow(reshape(x[1], n, m), cmap=c)
title("sparse")
axis("off")
subplot(1, 3, 3)
imshow(reshape(x[2], n, m), cmap=c)
title("sparse-in-frequency")
axis("off")
subplots_adjust(wspace=0.05, hspace=0)
 