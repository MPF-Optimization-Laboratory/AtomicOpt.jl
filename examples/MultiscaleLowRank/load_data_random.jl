using MAT
using AtomicOpt
using LinearAlgebra
using BlockArrays
using PyPlot
using JLD
file = matopen("./examples/MultiscaleLowRank/MultiscaleLowRankRandom.mat")
# X = read(file, "X")
X_decom = read(file, "X_decom")
close(file)

x1 = X_decom[:,:,1];
x2 = X_decom[:,:,2];
x3 = X_decom[:,:,3];
x4 = X_decom[:,:,4];

# first atomic set
n = 64
ks = count(x->x!=0, X_decom[:,:,1])
A1 = OneBall(n^2, maxrank=ks); A1 = gauge(A1, x1)*A1

# second atomic set
bn = 4; l = floor(Int, n/bn)
Xb2 = BlockArray(x2, repeat([bn], l), repeat([bn], l))
idx = CartesianIndices((l, l))
kr2 = map(i->rank(Xb2[Block(i[1], i[2])]), idx)
A2 = BlkNucBall(n, n, bn, bn, kr2); A2 = gauge(A2, x2)*A2

# third atomic set
bn = 16; l = floor(Int, n/bn)
Xb3 = BlockArray(x3, repeat([bn], l), repeat([bn], l))
idx = CartesianIndices((l, l))
kr3 = map(i->rank(Xb3[Block(i[1], i[2])]), idx)
A3 = BlkNucBall(n, n, bn, bn, kr3); A3 = gauge(A3, x3)*A3

# fourth atomic set
kr4 = rank(x4)
A4 = NucBall(n, n, maxrank=kr4); A4 = gauge(A4, x4)*A4

# atomic set
A = A1 × A2 × A3 × A4

# observation
b = x1 + x2 + x3 + x4

# Plot
fig, axes= subplots(nrows=1, ncols=5,figsize=(8,2))
subplot(151)
imshow(b)
axis("off")
subplot(152)
imshow(x1)
axis("off")
subplot(153)
imshow(x2)
axis("off")
subplot(154)
imshow(x3)
axis("off")
subplot(155)
imshow(x4)
axis("off")
subplots_adjust(wspace=0.05, hspace=0)

save("./examples/MultiscaleLowRank/MultiscaleLowRankRandom.jld", "x1", vec(x1), 
                                                                "x2", vec(x2), 
                                                                "x3", vec(x3),
                                                                "x4", vec(x4),
                                                                "b", vec(b),
                                                                "A", A)
 

