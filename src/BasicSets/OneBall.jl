########################################################################
# Atoms of the 1-norm ball.
########################################################################

"""
    OneBallAtom(n, z)

Atom in the OneBall of dimension n.
"""
mutable struct OneBallAtom{T1<:Int64, T2<:SparseVector{Float64, Int64}} <: AbstractAtom
    n::T1
    z::T2
    function OneBallAtom(z::SparseVector{Float64, Int64})
        n = z.n
        new{Int64, SparseVector{Float64, Int64}}(n, z)
    end
end

"""
Multiply a one-norm atom by a linear map.
"""
Base.:(*)(M::AbstractLinearOp, a::OneBallAtom{Int64, SparseVector{Float64, Int64}}) = M*a.z
LinearAlgebra.mul!(Ma::AbstractVector{Float64}, M::LinearOp, a::OneBallAtom{Int64, SparseVector{Float64, Int64}}) = mul!(Ma, M, a.z)
Base.vec(a::OneBallAtom{Int64, SparseVector{Float64, Int64}}) = a.z
Base.length(a::OneBallAtom{Int64, SparseVector{Float64, Int64}}) = a.n

########################################################################
# Atomic set for the 1-norm.
########################################################################

"""
    OneBall(n, maxrank=n)

Atomic set defined by 1-norm ball in `n` variables.
The atomic set takes two optional parameters:

`maxrank` is the maximum dimension of the face exposed by a vector.
"""
struct OneBall{T<:Int64} <: AbstractAtomicSet
    n::T
    maxrank::T
    function OneBall(n::Int64, maxrank::Int64)
        n ≥ 1 || throw(DomainError(n,"n must be a positive integer"))
        n ≥ maxrank ≥ 1 || throw(DomainError(maxrank,"maxrank must be ≥ 1"))
        new{Int64}(n, maxrank)
    end
end
OneBall(n::Int64; maxrank=n) = OneBall(n, maxrank)

"""
    gauge(A::OneBall, x::Vector)

Gives the 1-norm of `x`.
"""
gauge(::OneBall{Int64}, x::Vector{Float64}) = norm(x,1)
gauge(A::OneBall{Int64}, x::Matrix{Float64}) = gauge(A, vec(x))

"""
    support(A::OneBall, z::Vector)

Gives the inf-norm of `z`.
"""
support(::OneBall{Int64}, z::Vector{Float64}) = norm(z,Inf)

"""
    expose!(A::OneBall, z::Vector, a::OneBallAtom; tol=1e-2)

Obtain an atom in the face exposed by the vector `z`.
The atom `a` is overwritten. 
"""
function expose!(A::OneBall{Int64}, z::Vector{Float64}, a::OneBallAtom{Int64, SparseVector{Float64, Int64}}; tol=1e-2)
    n = A.n
    @inbounds a.z .= spzeros(n)
    zmax = support(A, z)
    if zmax ≥ 1e-12
        nnz = 0
        @inbounds for i in 1:n
            val = z[i]
            if zmax - abs(val) < tol*zmax
                nnz += 1
                push!(a.z.nzind, i)
                push!(a.z.nzval, sign(val))
            end
        end
        a.z.nzval ./= nnz
    end
    return nothing
end

function expose(A::OneBall{Int64}, z::Vector{Float64})
    a = OneBallAtom(spzeros(A.n))
    expose!(A, z, a)
    return a
end

Base.length(A::OneBall{Int64}) = A.n
rank(A::OneBall{Int64}) = A.maxrank
atom_name(A::OneBall{Int64}) = "one-norm ball"
atom_description(A::OneBall{Int64}) = "{ x ∈ ℝⁿ | ||x||₁ ≤ 1 }"
atom_parameters(A::OneBall{Int64}) = "n = $(length(A)); rank = $(A.maxrank)"


########################################################################
# Face of the one-norm ball.
########################################################################

mutable struct OneBallFace{T1<:Int64, T2<:SparseMatrixCSC{Float64, Int64}} <: AbstractFace
    n::T1
    k::T1
    S::T2
    function OneBallFace(S::SparseMatrixCSC{Float64, Int64}) 
        n, k = size(S)
        new{Int64, SparseMatrixCSC{Float64, Int64}}(n, k, S)
    end
end

"""
    face!(A, z, F)

(Inplace) Return a face of the atomic set `A` exposed by the vector `z`. The dimension of
the exposed face is equal to `k=rank(A)`. 
"""
function face!(A::OneBall{Int64}, z::Vector{Float64}, F::OneBallFace{Int64, SparseMatrixCSC{Float64, Int64}})
    k = A.maxrank
    idx = partialsortperm(abs.(z), 1:k, rev=true)
    val = [sign(z[i]) for i in idx]
    @inbounds F.S.rowval .= idx
    @inbounds F.S.nzval .= val
    return nothing
end

function face(A::OneBall{Int64}, z::Vector{Float64})
    n, k = A.n, A.maxrank
    F = OneBallFace( sparse(collect(1:k), collect(1:k), ones(k), n, k) )
    face!(A, z, F)
    return F
end


"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
Base.:(*)(F::OneBallFace{Int64, SparseMatrixCSC{Float64, Int64}}, c::Vector{Float64}) = F.S*c
Base.:(*)(M::AbstractLinearOp, F::OneBallFace{Int64, SparseMatrixCSC{Float64, Int64}}) = M*F.S
Base.:(*)(λ::Float64, F::OneBallFace{Int64, SparseMatrixCSC{Float64, Int64}}) = λ*F.S
Base.length(F::OneBallFace{Int64, SparseMatrixCSC{Float64, Int64}}) = F.n
rank(F::OneBallFace{Int64, SparseMatrixCSC{Float64, Int64}}) = F.k
vec(F::OneBallFace{Int64, SparseMatrixCSC{Float64, Int64}}) = F.S
face_name(F::OneBallFace{Int64, SparseMatrixCSC{Float64, Int64}}) = "one-norm ball"
face_parameters(F::OneBallFace{Int64, SparseMatrixCSC{Float64, Int64}}) = "rank = $(rank(F)); n = $(length(F))"
