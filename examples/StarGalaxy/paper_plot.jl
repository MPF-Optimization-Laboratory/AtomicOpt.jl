using LinearAlgebra
using PyPlot
using BSON: @save, @load

@load "./examples/StarGalaxy/result.bson" b x 
@load "./examples/StarGalaxy/data.bson" xs xd ks kd b
n, m = size(xs)

# plot
c = "gray"
fig, axes= subplots(nrows=1, ncols=3)
subplot(1, 3, 1)
imshow(reshape(b, n, m), cmap=c)
# title("observation")
axis("off")
subplot(1, 3, 2)
imshow(reshape(x[1], n, m), cmap=c)
# title("sparse")
axis("off")
subplot(1, 3, 3)
imshow(reshape(x[2], n, m), cmap=c)
# title("sparse-in-frequency")
axis("off")
subplots_adjust(wspace=0.05, hspace=0)