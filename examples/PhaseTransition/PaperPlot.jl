using BSON: @save, @load
using PyPlot
using LaTeXStrings
using Optim
using QuadGK
using Statistics

@load "./examples/PhaseTransition/NoisyMeasurement.bson" E
@load "./examples/PhaseTransition/NoiselessMeasurement2.bson" E1000
@load "./examples/PhaseTransition/NoiselessMeasurement500.bson" E3

g(τ) = quadgk(u->(u-τ)^2*sqrt(2/π)*exp(-u^2/2), τ, Inf, rtol=1e-3)[1]
function δ(n, s) 
    ρ = s/n
    f(τ) = ρ*(1+τ^2) + (1 - ρ)*g(τ)
    res = optimize(f, 0.0, 1000.0)
    return minimum(res)*n
end


fig, axes = subplots(nrows=1, ncols=1, figsize=(5,5))

# subplot(1,1,1)
# Δ(k, n, s) = max(100, k^2 * (δ(n, s) - n/sqrt(n*s)) );
# tol = 1e-2;
# s = 5;
# ns = collect(50:15:500)
# ticks = [50, 200, 350, 500]
# c = 1.0
# P3 = (E3 .< tol); S3 = sum(P3, dims=3)[:,:,1] ./ 50
# im = imshow(S3, cmap="gray", extent=[50,500,50,500], vmin=0, vmax=1)
# k = 3; ms = map(n->Δ(k,n,s), ns); plot(ns, ms, color="red")
# nl = 50 + mapreduce(x->x<0,+,ns - c*ms)*15
# plot(repeat([nl], 31), ns, color="blue", linestyle="dashed")
# xticks(ticks, fontsize=8)
# yticks(ticks, fontsize=8)
# xlabel("n")
# ylabel("m")
# cb_ax = fig.add_axes([1.55, 0.1, 0.02, 0.8])
# fig.colorbar(im, cax=cb_ax) 
# savefig("./examples/PhaseTransition/ralation_m_n.pdf")


# subplot(1,1,1)
# tol = 1e-2;
# Δ(k, n, s) = max(100, k^2 * (δ(n, s) - 0.8*n/sqrt(n*s)) );
# P = (E1000 .< tol); S = sum(P, dims=3)[:,:,1] ./ 10
# im = imshow(S, cmap="gray", vmin=0, vmax=1, extent=[2, 10, 100, 1000], aspect=0.01)
# ks = collect(2:10); n=1000; s=3
# ms = map(k->Δ(k,n,s), ks)
# plot(ks, ms, color="red")
# xticks(collect(2:10), fontsize=8)
# yticks(100*collect(1:10), fontsize=8)
# xlabel("k")
# ylabel("m")
# cb_ax = fig.add_axes([1.55, 0.1, 0.02, 0.8])
# fig.colorbar(im, cax=cb_ax) 
# savefig("./examples/PhaseTransition/ralation_m_k.pdf")


subplot(1,1,1)
αs = 0.01 .* collect(1:200)
avg = sum(E, dims=2) ./ 50
s= zeros(200)
for i in 1:200
    s[i] = std(E[i, :],  mean=avg[i])
end
upper = avg + s; lower = avg - s
axes.plot(αs, avg, color="blue")
axes.fill_between(αs, upper[:,1], lower[:,1], facecolor="yellow", alpha=0.5)
xlim([0.0, 2.0])
ylim([0.0, 0.25])
# xticks(collect(2:10), fontsize=8)
# yticks(100*collect(1:10), fontsize=8)
xlabel("noise level, α")
ylabel("maximum absolute error")
savefig("./examples/PhaseTransition/ralation_e_noise.pdf")
 

display(gcf())

