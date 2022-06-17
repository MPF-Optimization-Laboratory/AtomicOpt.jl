########################################################################
# Atoms of the block nuclear-norm ball.
########################################################################
"""
    BlkNucBallAtom(m,n,cidx,u,v)

Atom in the BlkNucBall with index cidx. 
"""
mutable struct BlkNucBallAtom{T1<:Int64, T2<:CartesianIndex, T3<:NucBallAtom{Int64, Vector{Float64}}} <: AbstractAtom
    m::T1
    n::T1
    cidx::T2
    ba::T3
    function BlkNucBallAtom(m::Int64, n::Int64, cidx::CartesianIndex, ba::NucBallAtom{Int64, Vector{Float64}})
        new{Int64, CartesianIndex, NucBallAtom{Int64, Vector{Float64}}}(m, n, cidx, ba)
    end
end

"""
    *(M::AbstractLinearOp, a::NucBallAtom) -> Y

Multiply a nuclear-norm atom by a linear map, and return the matrix `Y`.
"""
function Base.:(*)(M::AbstractLinearOp, a::BlkNucBallAtom)
    m, n = a.m, a.n
    c1, c2 = a.cidx[1], a.cidx[2]
    u, v = a.ba.u, a.ba.v
    bm = length(u); bn = length(v)
    A = spzeros(m*n)
    @inbounds for bj in 1:bn
        @inbounds for bi in 1:bm
            i = (c1-1)*bm + bi
            j = (c2-1)*bn + bj
            idx = (j-1)*m + i
            val = u[bi] * v[bj]
            push!(A.nzind, idx)
            push!(A.nzval, val)
        end
    end
    return M*A
end
function LinearAlgebra.mul!(Ma::Vector{Float64}, M::LinearOp, a::BlkNucBallAtom)
    m, n = a.m, a.n
    c1, c2 = a.cidx[1], a.cidx[2]
    u, v = a.ba.u, a.ba.v
    bm = length(u); bn = length(v)
    A = spzeros(m*n)
    @inbounds for bj in 1:bn
        @inbounds for bi in 1:bm
            i = (c1-1)*bm + bi
            j = (c2-1)*bn + bj
            idx = (j-1)*m + i
            val = u[bi] * v[bj]
            push!(A.nzind, idx)
            push!(A.nzval, val)
        end
    end
    mul!(Ma, M, A)
    return nothing
end
Base.vec(a::BlkNucBallAtom) = I(a.m*a.n)*a
Base.length(a::BlkNucBallAtom) = a.m * a.n

########################################################################
# Block Nuclear-norm ball.
########################################################################
"""
    BlkNucBall(m, n, bm, bn, maxrank)

"""
struct BlkNucBall{T1<:Int64, T2<:Matrix{Int64}} <: AbstractAtomicSet
    m::T1           # matrix size 1
    n::T1           # matrix size 2
    bm::T1          # block size 1
    bn::T1          # block size 2
    l::T1           # num of blocks 1
    h::T1           # num of blocks 2
    maxrank::T2     # max rank for each block
    function BlkNucBall(n::Int64, m::Int64, bn::Int64, bm::Int64, maxrank::Matrix{Int64} = ones(Int64, floor(Int, m/bm), floor(Int, n/bn)))
        l = floor(Int, m/bm)
        h = floor(Int, n/bn)
        new{Int64, Matrix{Int64}}(m, n, bm, bn, l, h, maxrank)
    end
end


"""
    gauge(A::BlkNucBall, x::Matrix)

Gives the block nuclear-norm of `x`.
"""
function gauge(A::BlkNucBall, x::Matrix{Float64}) 
    bm, bn, l, h = A.bm, A.bn, A.l, A.h
    Ab = NucBall(bm, bn)
    f(i::CartesianIndex) = gauge(Ab, x[(i[1]-1)*bm+1:i[1]*bm, (i[2]-1)*bn+1:i[2]*bn])
    return mapreduce(f, +, CartesianIndices((l, h)))
end
gauge(A::BlkNucBall, x::Vector{Float64}) = gauge(A, reshape(x, A.m, A.n))


"""
    support(A::BlkNucBall, z::Matirx)

Gives the largest singular value of `z`.
"""
function support(A::BlkNucBall, z::Matrix{Float64})
    bm, bn, l, h = A.bm, A.bn, A.l, A.h
    Ab = NucBall(bm, bn)
    f(i::CartesianIndex) = support(Ab, z[(i[1]-1)*bm+1:i[1]*bm, (i[2]-1)*bn+1:i[2]*bn])
    return maximum(pmap(f, CartesianIndices((l, h))))
end
support(A::BlkNucBall, z::Vector{Float64}) = support(A, reshape(z, A.m, A.n))


"""
    expose(A::BlkNucBall, z::Matrix)

"""
function expose!(A::BlkNucBall, z::Matrix{Float64}, a::BlkNucBallAtom)
    bm, bn, l, h = A.bm, A.bn, A.l, A.h
    Ab = NucBall(bm, bn)
    f(i::CartesianIndex) = support(Ab, z[(i[1]-1)*bm+1:i[1]*bm, (i[2]-1)*bn+1:i[2]*bn])
    ci = argmax(pmap(f, CartesianIndices((l, h))))
    a.cidx = ci
    expose!(Ab, z[(ci[1]-1)*bm+1:ci[1]*bm, (ci[2]-1)*bn+1:ci[2]*bn], a.ba)
    return nothing
end
expose!(A::BlkNucBall, z::Vector{Float64}, a::BlkNucBallAtom) = expose!(A, reshape(z, A.m, A.n), a)

function expose(A::BlkNucBall, z::Matrix{Float64})
    bm, bn = A.bm, A.bn
    Ab = NucBall(bm, bn)
    ba = expose(Ab, zeros(bm, bn))
    a = BlkNucBallAtom(A.m, A.n, CartesianIndex(1, 1), ba)
    expose!(A, z, a)
    return a
end
expose(A::BlkNucBall, z::Vector{Float64}) = expose(A, reshape(z, A.m, A.n))


Base.length(A::BlkNucBall) = A.m*A.n
rank(A::BlkNucBall) = A.maxrank
atom_name(A::BlkNucBall) = "block nuclear-norm ball"
atom_description(A::BlkNucBall) = "{ x ∈ ℝⁿ | ||x||₁ ≤ 1 }"
atom_parameters(A::BlkNucBall) = "n×m = $(length(A)); maxrank = $(A.maxrank)"

########################################################################
# Face of the nuclear-norm ball.
########################################################################
mutable struct BlkNucBallFace{T1<:Int64, T2<:Matrix{NucBallFace{Int64, Matrix{Float64}, Adjoint{Float64, Matrix{Float64}}}}, T3<:SparseMatrixCSC{Float64, Int64}} <: AbstractFace
    m::T1
    n::T1
    nucBallFaces::T2
    S::T3
    function BlkNucBallFace(m::Int64, n::Int64, nucBallFaces::Matrix{NucBallFace{Int64, Matrix{Float64}, Adjoint{Float64, Matrix{Float64}}}})
        l, h = size(nucBallFaces)
        bm, bn = size(nucBallFaces[1,1])
        I = Vector{Int64}(undef, m*n)
        for k in 1:m*n
            I[k] = indexmatch(k, m, bm, l, bn)
        end
        S = sparse(I, collect(1:m*n), ones(m*n), m*n, m*n)
        new{Int64, Matrix{NucBallFace{Int64, Matrix{Float64}, Adjoint{Float64, Matrix{Float64}}}},SparseMatrixCSC{Float64, Int64}}(m, n, nucBallFaces, S)
    end
end

function indexmatch(k::Int64, m::Int64, bm::Int64, l::Int64, bn::Int64)
    u, v = divrem(k, bm*bn)
    if v == 0
        u = u - 1; v = bm*bn
    end
    c, d = divrem(u+1, l)
    if d == 0
        c = c - 1; d = l
    end
    e, f = divrem(v, bm)
    if f == 0
        e = e - 1; f = bm
    end
    return (c*bn + e)*m + (d-1)*bm + f
end



"""
    face(A, z) -> F

"""
function face!(A::BlkNucBall, z::Matrix{Float64}, F::BlkNucBallFace)
    bm, bn, l, h, maxrank = A.bm, A.bn, A.l, A.h, A.maxrank
    f(i::CartesianIndex) = face!( NucBall(bn, bm, maxrank=maxrank[i]), z[(i[1]-1)*bm+1:i[1]*bm, (i[2]-1)*bn+1:i[2]*bn], F.nucBallFaces[i])
    pmap(f, CartesianIndices((l, h)))
    return nothing
end
face!(A::BlkNucBall, z::Vector{Float64}, F::BlkNucBallFace) = face!(A, reshape(z, A.m, A.n), F)

function face(A::BlkNucBall, z::Matrix{Float64})
    bm, bn, l, h, maxrank = A.bm, A.bn, A.l, A.h, A.maxrank
    f(i::CartesianIndex) = face( NucBall(bn, bm, maxrank=maxrank[i]), z[(i[1]-1)*bm+1:i[1]*bm, (i[2]-1)*bn+1:i[2]*bn])
    Fs = pmap(f, CartesianIndices((l, h)))
    return BlkNucBallFace(A.m, A.n, Fs)
end
face(A::BlkNucBall, z::Vector{Float64}) = face(A, reshape(z, A.m, A.n))

"""
    *(M::AbstractLinearOp, F::NucBallFace)
"""
function Base.:(*)(M::AbstractLinearOp, F::BlkNucBallFace) 
    return M*F.S*Base.cat(map(vec, F.nucBallFaces)...; dims=(1,2))
end
Base.:(*)(λ::Real, F::BlkNucBallFace) = λ*vec(F)
Base.length(F::BlkNucBallFace) = F.n*F.m
rank(F::BlkNucBallFace) = sum(map(rank, F.nucBallFaces))
vec(F::BlkNucBallFace) = I(F.m*F.n)*F
face_name(F::BlkNucBallFace) = "block nuclear-norm ball"
face_parameters(F::BlkNucBallFace) = "rank = $(rank(F)); n = $(length(F))"


"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
Base.:(*)(F::BlkNucBallFace, c::Vector) = vec(F)*c