using BSON: @save, @load
using PyPlot
using LaTeXStrings
using Optim
using QuadGK

@load "./examples/PhaseTransition/NoisyMeasurement.bson" E

g(τ) = quadgk(u->(u-τ)^2*sqrt(2/π)*exp(-u^2/2), τ, Inf, rtol=1e-3)[1]
function δ(n, s) 
    ρ = s/n
    f(τ) = ρ*(1+τ^2) + (1 - ρ)*g(τ)
    res = optimize(f, 0.0, 1000.0)
    return (minimum(res) - 2/sqrt(n*s))*n
end
# δ(n, s) = s*log(n/s);
Δ(k, n, s) = min(max(50, k^2*δ(n, s)), 200);
s = 5;
ns = collect(50:15:200)
ticks = [50, 100, 150, 200]
c = 0.5

fig, axes = subplots(nrows=1, ncols=3)
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, wspace=0.27, hspace=0.02)

tol = 0.001;

subplot(1,3,1)
P2 = (E2 .< tol); S2 = sum(P2, dims=3)[:,:,1] ./ 50
im = imshow(S2, cmap="Blues", extent=[50,200,50,200], vmin=0, vmax=1)
k = 2; ms = map(n->Δ(k,n,s), ns); plot(ns, ms, color="red")
nl = 50 + mapreduce(x->x<0,+,ns - c*ms)*15
plot(repeat([nl], 11), ns, color="yellow", linestyle="dashed")
xticks(ticks, fontsize=6)
yticks(ticks, fontsize=6)
xlabel("n")
ylabel("m")
title("k=2")

subplot(1,3,2)
P3 = (E3 .< tol); S3 = sum(P3, dims=3)[:,:,1] ./ 50
im = imshow(S3, cmap="Blues", extent=[50,200,50,200], vmin=0, vmax=1)
k = 3; ms = map(n->Δ(k,n,s), ns); plot(ns, ms, color="red")
nl = 50 + mapreduce(x->x<0,+,ns - c*ms)*15
plot(repeat([nl], 11), ns, color="yellow", linestyle="dashed")
xticks(ticks, fontsize=6)
yticks(ticks, fontsize=6)
xlabel("n")
# ylabel("m")
title("k=3")

subplot(1,3,3)
P4 = (E4 .< tol); S4 = sum(P4, dims=3)[:,:,1] ./ 50
k = 4; ms = map(n->Δ(k,n,s), ns)
im = imshow(S4, cmap="Blues", extent=[50,200,50,200], vmin=0, vmax=1)
plot(ns, ms, color="red")
nl = 50 + mapreduce(x->x<0,+,ns - c*ms)*15
plot(repeat([nl], 11), ns, color="yellow", linestyle="dashed")
xticks(ticks, fontsize=6)
yticks(ticks, fontsize=6)
xlabel("n")
# ylabel("m")
title("k=4")


cb_ax = fig.add_axes([0.9, 0.36, 0.02, 0.28])
fig.colorbar(im, cax=cb_ax) 