using PyPlot
using BSON: @save, @load

@load "/Users/zhenanfan/.julia/dev/PolarDemixing/examples/ChessBoard/noisy2.bson" xsp xrp bp up

plots = [bp, abs.(xsp), xrp, bp-up]
c = "gray"
for i = 1:4
    subplot(2,2,i)
    imshow(plots[i], cmap=c)
    axis("off")
end
tight_layout(w_pad=-11, h_pad=-2, pad=0)