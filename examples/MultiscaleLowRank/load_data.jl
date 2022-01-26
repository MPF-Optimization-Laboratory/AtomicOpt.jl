using MAT
using LinearAlgebra
using BlockArrays

file = matopen("./examples/MultiscaleLowRank/MultiscaleLowRankRandom.mat")
X_decom = read(file, "X_decom")
close(file)

x1 = X_decom[:,:,1]
x2 = X_decom[:,:,2]
x3 = X_decom[:,:,3]
x4 = X_decom[:,:,4]
m, n = size(x1)

# first component
n = 64
kr1 = count(x->x!=0, x1)
x1 = vec(x1)

# second component
bn2 = 4; l = floor(Int, n/bn2)
Xb2 = BlockArray(x2, repeat([bn2], l), repeat([bn2], l))
idx2 = CartesianIndices((l, l))
kr2 = map(i->rank(Xb2[Block(i[1], i[2])]), idx2)
kr2 = vec(kr2)
x2 = vec(x2)

# third component
bn3 = 16; l = floor(Int, n/bn3)
Xb3 = BlockArray(x3, repeat([bn3], l), repeat([bn3], l))
idx3 = CartesianIndices((l, l))
kr3 = map(i->rank(Xb3[Block(i[1], i[2])]), idx3)
kr3 = vec(kr3)
x3 = vec(x3)

# fourth component
kr4 = rank(x4)
x4 = vec(x4)


# observation
b = x1 + x2 + x3 + x4

# write to file
f = open("/home/zhenan/Github/AtomicOpt.jl/examples/MultiscaleLowRank/MultiscaleData.txt", "w")
l = length(b)
@printf(f, "x1 = ")
for i in 1:l
    @printf(f, "%.3f ", x1[i])
end
println(f, "\n")
@printf(f, "x2 = ")
for i in 1:l
    @printf(f, "%.3f ", x2[i])
end
println(f, "\n")
@printf(f, "x3 = ")
for i in 1:l
    @printf(f, "%.3f ", x3[i])
end
println(f, "\n")
@printf(f, "x4 = ")
for i in 1:l
    @printf(f, "%.3f ", x4[i])
end
println(f, "\n")
@printf(f, "b = ")
for i in 1:l
    @printf(f, "%.3f ", b[i])
end
println(f, "\n")
@printf(f, "kr1 = %d\n", kr1)
@printf(f, "kr2 = ")
for t in kr2
    @printf(f, "%d ", t)
end
println(f, "\n")
@printf(f, "kr3 = ")
for t in kr3
    @printf(f, "%d ", t)
end
println(f, "\n")
@printf(f, "kr4 = %d\n", kr4)
@printf(f, "m = %d\n", m)
@printf(f, "n = %d\n", n)
close(f)
 

