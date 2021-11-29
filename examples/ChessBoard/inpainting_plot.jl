using PyPlot
using BSON: @save, @load

@load "./examples/ChessBoard/noisy2.bson" xsp xrp bp up

plots = [bp, abs.(xsp), xrp, bp-up]
fig, axes= subplots(nrows=1, ncols=3)
c = "gray"
for i = 1:3
    subplot(1,3,i)
    imshow(plots[i], cmap=c)
    axis("off")
end
tight_layout(w_pad=1, h_pad=-1, pad=0)