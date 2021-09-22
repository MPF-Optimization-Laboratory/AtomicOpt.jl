########################################################################
# Atoms of the block nuclear-norm ball.
########################################################################
"""
    NucBallAtom(n, m, u, v)

Atom in the NucBall of dimension n×m.
"""
struct BlkNucBallAtom <: AbstractAtom
    n::Int64
    m::Int64
    cidx::CartesianIndex
    ba::NucBallAtom
end

"""
    *(M::AbstractLinearOp, a::NucBallAtom) -> Y

Multiply a nuclear-norm atom by a linear map, and return the matrix `Y`.
"""
function Base.:(*)(M::AbstractLinearOp, a::BlkNucBallAtom)
    bn, bm = size(a.ba)
    l, h = floor(Int, a.n/bn), floor(Int, a.m/bm)
    i, j = a.cidx[1], a.cidx[2]
    A = zeros(a.n, a.m)
    A[(i-1)*bn+1:i*bn, (j-1)*bm+1:j*bm] .= a.ba.u*a.ba.v'
    return M*vec(A)
end
Base.vec(a::BlkNucBallAtom) = I(a.n*a.m)*a
Base.length(a::BlkNucBallAtom) = a.n * a.m

########################################################################
# Block Nuclear-norm ball.
########################################################################
"""
    NucBall(n, m, maxrank=n)

Atomic set defined by nuclear-norm ball for the space of `m`-by-`n` matrices.
The atomic set takes two optional parameters:

`maxrank` is the maximum dimension of the face exposed by a vector.
"""
struct BlkNucBall <: AbstractAtomicSet
    n::Int64
    m::Int64
    bn::Int64
    bm::Int64
    l::Int64
    h::Int64
    maxrank::Matrix{Int64}
    Idx1::Vector{Int64}
    Idx2::Vector{Int64}
    function BlkNucBall(n, m, bn, bm, maxrank)
        l = floor(Int, n/bn)
        h = floor(Int, m/bm)
        g(a) = (indexmatch(a,n,m,bn,bm,l,h), a)
        dict = Dict(map(g, collect(1:n*m)))
        Idx1 = map(k->indexmatch(k,n,m,bn,bm,l,h), collect(1:n*m))
        Idx2 = map(k->dict[k], collect(1:n*m))
        new(n, m, bn, bm, l, h, maxrank, Idx1, Idx2)
    end
end


"""
    gauge(A::BlkNucBall, x::Matrix)

Gives the block nuclear-norm of `x`.
"""
function gauge(A::BlkNucBall, x::Matrix) 
    n, m, bn, bm, l, h = A.n, A.m, A.bn, A.bm, A.l, A.h
    f(i) = gauge(NucBall(bn, bm), x[(i[1]-1)*bn+1:i[1]*bn, (i[2]-1)*bm+1:i[2]*bm])
    return mapreduce(f, +, CartesianIndices((l, h)))
end
gauge(A::BlkNucBall, x::Vector) = gauge(A, reshape(x, A.n, A.m))


"""
    support(A::BlkNucBall, z::Matirx)

Gives the largest singular value of `z`.
"""
function support(A::BlkNucBall, z::Matrix)
    n, m, bn, bm, l, h = A.n, A.m, A.bn, A.bm, A.l, A.h
    f(i) = support(NucBall(bn, bm), z[(i[1]-1)*bn+1:i[1]*bn, (i[2]-1)*bm+1:i[2]*bm])
    return maximum(pmap(f, CartesianIndices((l, h))))
end

support(A::BlkNucBall, z::Vector) = support(A, reshape(z, A.n, A.m))


"""
    expose(A::BlkNucBall, z::Matrix)

"""
function expose(A::BlkNucBall, z::Matrix)
    n, m, bn, bm, l, h = A.n, A.m, A.bn, A.bm, A.l, A.h
    f(i) = support(NucBall(bn, bm), z[(i[1]-1)*bn+1:i[1]*bn, (i[2]-1)*bm+1:i[2]*bm])
    mi = argmax(pmap(f, CartesianIndices((l, h))))
    ba = expose(NucBall(bn, bm), z[(mi[1]-1)*bn+1:mi[1]*bn, (mi[2]-1)*bm+1:mi[2]*bm])
    return BlkNucBallAtom(n, m, mi, ba)
end

expose(A::BlkNucBall, z::Vector) = expose(A, reshape(z, A.n, A.m))

Base.length(A::BlkNucBall) = A.n*A.m
rank(A::BlkNucBall) = A.maxrank
atom_name(A::BlkNucBall) = "block nuclear-norm ball"
atom_description(A::BlkNucBall) = "{ x ∈ ℝⁿ | ||x||₁ ≤ 1 }"
atom_parameters(A::BlkNucBall) = "n×m = $(length(A)); maxrank = $(A.maxrank)"

########################################################################
# Face of the nuclear-norm ball.
########################################################################
struct BlkNucBallFace <: AbstractFace
    n::Int64
    m::Int64
    nucfaces::Matrix{NucBallFace}
    IM::LinearMap{Float64}
end

function indexmatch(k::Int64, n::Int64, m::Int64, bn::Int64, bm::Int64, l::Int64, h::Int64)
    u, v = divrem(k,bn*bm)
    if v == 0
        u = u - 1; v = bn*bm
    end
    c, d = divrem(u+1, l)
    if d == 0
        c = c - 1; d = l
    end
    e, f = divrem(v, bn)
    if f == 0
        e = e - 1; f = bn
    end
    return (c*bm + e)*n + (d-1)*bn + f
end

"""
    face(A, z) -> F

Return a face `F` of the atomic set `A` exposed by the vector `z`. The dimension of
the exposed face is limited by `k=maxrank(A)`. If the dimension is being limited
by `k`, then the routine returns a set of rank `k` atoms that define the subset
of the exposed face.
"""
function face(A::BlkNucBall, z::Matrix; rTol=0.1)
    n, m, bn, bm, l, h, maxrank = A.n, A.m, A.bn, A.bm, A.l, A.h, A.maxrank
    f(i) = face(NucBall(bn, bm, maxrank=maxrank[i]), z[(i[1]-1)*bn+1:i[1]*bn, (i[2]-1)*bm+1:i[2]*bm])
    nucfaces = map(f, CartesianIndices((l, h)))
    IM = LinearMap(y->y[A.Idx2], y->y[A.Idx1], n*m, n*m)
    return BlkNucBallFace(n, m, nucfaces, IM)
end

face(A::BlkNucBall, z::Vector; kwargs...) = face(A, reshape(z, A.n, A.m); kwargs...)

"""
    *(M::AbstractLinearOp, F::NucBallFace)
"""
function Base.:(*)(M::AbstractLinearOp, F::BlkNucBallFace) 
    return M*F.IM*Base.cat(map(vec, F.nucfaces)...; dims=(1,2))
end
Base.:(*)(λ::Real, F::BlkNucBallFace) = λ*vec(F)
Base.length(F::BlkNucBallFace) = F.n*F.m
rank(F::BlkNucBallFace) = map(rank, F.nucfaces)
vec(F::BlkNucBallFace) = I(F.m*F.n)*F
face_name(F::BlkNucBallFace) = "block nuclear-norm ball"
face_parameters(F::BlkNucBallFace) = "rank = $(rank(F)); n = $(length(F))"


"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
Base.:(*)(F::BlkNucBallFace, c::Vector) = vec(F)*c