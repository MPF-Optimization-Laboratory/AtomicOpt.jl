using BSON: @save, @load
using PyPlot
using LaTeXStrings
using Optim
using QuadGK

@load "./Measurement_k=2.bson" E2
@load "./Measurement_k=3.bson" E3
@load "./Measurement_k=4.bson" E4
@load "./Measurement_k=5.bson" E5
@load "./Measurement_k=6.bson" E6
@load "./Measurement_k=7.bson" E7

g(τ) = quadgk(u->(u-τ)^2*sqrt(2/π)*exp(-u^2/2), τ, Inf, rtol=1e-3)[1]
function δ(n, s) 
    ρ = s/n
    f(τ) = ρ*(1+τ^2) + (1 - ρ)*g(τ)
    res = optimize(f, 0.0, 1000.0)
    return (minimum(res) - 1/sqrt(n*s))*n
end
# δ(n, s) = s*log(n/s);
Δ(k, n, s) = max(50, k^2*δ(n, s));
s = 5;
ns = collect(50:15:500)
ticks = [50, 200, 350, 500]
c = 0.5
vx = 0.1

fig, axes = subplots(nrows=1, ncols=3)
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, wspace=0.27, hspace=0.02)

subplot(1,3,1)
im = imshow(0.7*E2, cmap="Blues", extent=[50,500,50,500], vmin=0, vmax=vx)
k = 2; ms = map(n->Δ(k,n,s), ns); plot(ns, ms, color="red")
nl = 50 + mapreduce(x->x<0,+,ns - c*ms)*15
plot(repeat([nl], 31), ns, color="yellow", linestyle="dashed")
xticks(ticks, fontsize=6)
yticks(ticks, fontsize=6)
xlabel("n")
ylabel("m")
title("k=2")

subplot(1,3,2)
im = imshow(E4, cmap="Blues", extent=[50,500,50,500], vmin=0, vmax=vx)
k = 3; ms = map(n->Δ(k,n,s), ns); plot(ns, ms, color="red")
nl = 50 + mapreduce(x->x<0,+,ns - c*ms)*15
plot(repeat([nl], 31), ns, color="yellow", linestyle="dashed")
xticks(ticks, fontsize=6)
yticks(ticks, fontsize=6)
xlabel("n")
# ylabel("m")
title("k=3")

subplot(1,3,3)
k = 4; ms = map(n->Δ(k,n,s), ns)
im = imshow(E6, cmap="Blues", extent=[50,500,50,500], vmin=0, vmax=vx)
plot(ns, ms, color="red")
nl = 50 + mapreduce(x->x<0,+,ns - c*ms)*15
plot(repeat([nl], 31), ns, color="yellow", linestyle="dashed")
xticks(ticks, fontsize=6)
yticks(ticks, fontsize=6)
xlabel("n")
# ylabel("m")
title("k=4")

# subplot(2,3,4)
# im = imshow(E5, cmap="Blues", extent=[50,500,50,500], vmin=0, vmax=0.1)
# k = 5; ms = map(n->Δ(k,n,s), ns); plot(ns, ms, color="red")
# nl = 50 + mapreduce(x->x<0,+,ns - c*ms)*15
# plot(repeat([nl], 31), ns, color="yellow", linestyle="dashed")
# xticks(ticks, fontsize=6)
# yticks(ticks, fontsize=6)
# title("k=5")

# subplot(2,3,5)
# im = imshow(E6, cmap="Blues", extent=[50,500,50,500], vmin=0, vmax=0.1)
# k = 6; ms = map(n->Δ(k,n,s), ns); plot(ns, ms, color="red")
# nl = 50 + mapreduce(x->x<0,+,ns - c*ms)*15
# plot(repeat([nl], 31), ns, color="yellow", linestyle="dashed")
# xticks(ticks, fontsize=6)
# yticks(ticks, fontsize=6)
# title("k=6")

# subplot(2,3,6)
# im = imshow(E7, cmap="Blues", extent=[50,500,50,500], vmin=0, vmax=0.1)
# k = 7; ms = map(n->Δ(k,n,s), ns); plot(ns, ms, color="red")
# nl = 50 + mapreduce(x->x<0,+,ns - c*ms)*15
# plot(repeat([nl], 31), ns, color="yellow", linestyle="dashed")
# xticks(ticks, fontsize=6)
# yticks(ticks, fontsize=6)
# title("k=7")

cb_ax = fig.add_axes([0.9, 0.36, 0.02, 0.28])
fig.colorbar(im, cax=cb_ax) 