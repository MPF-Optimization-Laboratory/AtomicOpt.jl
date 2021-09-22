########################################################################
# Atoms of the Positive Simplex
########################################################################

"""
    PosSimplexAtom(n, z)

Atom in the positive simplex of dimension n.
"""
struct PosSimplexAtom <: AbstractAtom
    n::Int64
    z::Vector{Float64}
    function PosSimplexAtom(z)
        n = length(z)
        return new(n, z)
    end
end

"""
Multiply a Positive simplex atom by a linear map.
"""
Base.:(*)(M::AbstractLinearOp, a::PosSimplexAtom) = M*a.z
Base.vec(a::PosSimplexAtom) = a.z
Base.length(a::PosSimplexAtom) = a.n

########################################################################
# Atomic set for the positive simplex 
########################################################################

"""
    PosSimplex(c, n, maxrank=n)

Atomic set defined by positive simplex in `n` variables.
The atomic set takes an optional parameters:

`maxrank` is the maximum dimension of the face exposed by a vector.
"""
struct PosSimplex <: AbstractAtomicSet
    c::Vector{Float64}
    n::Int64
    maxrank::Int64
    function PosSimplex(c, maxrank)
        n = length(c)
        minimum(c) > 0 || throw(DomainError(c,"c must be in non-negative orthant"))
        n ≥ maxrank ≥ 1 || throw(DomainError(maxrank,"maxrank must be ≥ 1"))
        return new(c, n, maxrank)
    end
end


function PosSimplex(c::Vector{Float64}; maxrank = length(findall(!iszero,c)) ) 
    index_c = findall(x -> x != 0, c)
    maxrank <= length(index_c) || throw(DomainError(maxrank,"maxrank should be less than sparstiy level of c"))
    return PosSimplex(c, maxrank)
end

PosSimplex(n::Int64; maxrank=n) = PosSimplex(ones(n), maxrank)


"""
    gauge(A::PosSimplex, x::Vector)

Gives the sum `dot(c, x)`.
"""
function gauge(A::PosSimplex, x::Vector) 
    for i in eachindex(x)
        if (A.c[i] == 0 && x[i] != 0) || x[i] < 0
            return Inf
        end
    end
    return dot(A.c,x)
end
"""
    support(A::PosSimplex, z::Vector)

Gives the inf-norm of `c ⊙ z`.
"""
function support(A::PosSimplex, z::Vector) 
    sup = 0. 
    for i in eachindex(z)
        if A.c[i] != 0
            sup = max(sup, z[i]/A.c[i])
        end
    end

    return sup
end
"""
    expose(A::PosSimplex, z::Vector)

A non-overwriting version of [`expose!`](@ref).
"""
expose(A::PosSimplex, z; kwargs...) = expose!(A::PosSimplex, copy(z); kwargs...)

"""
    expose!(A::PosSimplex, z::Vector; tol=1e-12)

Obtain an atom in the face exposed by the vector `z`.
The vector `z` is overwritten. If `norm(z,Inf)<tol`, then
`z` is returned untouched.
"""

function expose!(A::PosSimplex, z::Vector; tol=1e-1)
    # z = Float64.(z)
    zsup = support(A, z)
    if maximum(z) < -1e-12
        return PosSimplexAtom(zero(z))
    end

    index_c = findall(x -> x != 0, A.c) 
    index_c_comp = findall(x -> x == 0, A.c) 
    
    if zsup < 1e-12
        nnz = 1
    else
        nnz = 0
    end

    # for i in eachindex(index_c_comp)
    #     z[index_c_comp[i]] = 0
    # end
    for i in eachindex(index_c)
        val = (A.c[index_c[i]]).^(-1) * z[index_c[i]]
        if abs(zsup - val) ≤ tol*zsup
            z[index_c[i]] = 1/A.c[index_c[i]]
            nnz += 1
        else
            z[index_c[i]] = zero(eltype(z))
        end
    end
    return PosSimplexAtom(z ./= nnz)
end

Base.length(A::PosSimplex) = A.n
rank(A::PosSimplex) = A.maxrank
atom_name(A::PosSimplex) = "Positive Simplex"
atom_description(A::PosSimplex) = "{ x ∈ ℝⁿ | ⟨c, x⟩ ≤ 1, x ≥ 0}"
atom_parameters(A::PosSimplex) = "c = cost vector, n = $(length(A)); maxrank = $(A.maxrank)"

########################################################################
# Face of the positive simplex.
########################################################################

struct PosSimplexFace{LM <: LinearMap} <: AbstractFace
    n::Int64
    k::Int64
    S::LM
end

PosSimplexFace(S::LinearMap) = PosSimplexFace(size(S)...,S)

"""
    face(A, z)

Return a face of the atomic set `A` exposed by the vector `z`. The dimension of
the exposed face is limited by `k=maxrank(A)`. If the dimension is being limited
by `k`, then the routine returns a set of rank `k` atoms that define the subset
of the exposed face.
"""
function face(A::PosSimplex, z::Vector; rTol=1e-1)

    t = support(A, z)
    a = expose(A, z, tol=rTol*t)
    idx = findall(!iszero, vec(a))
    
    k = length(idx)
    val = [(A.c[i]).^(-1) for i in idx]
    n = length(A)
    if support(A,z) < 1e-12
        S = sparse(idx, collect(1:k), val, n, k + 1)
    else
        S = sparse(idx, collect(1:k), val, n, k)
    end
    S = LinearMap(S)
    return PosSimplexFace(S)
end

"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
Base.:(*)(F::PosSimplexFace, c::Vector) = F.S*c
Base.:(*)(M::AbstractLinearOp, F::PosSimplexFace) = M*F.S
Base.:(*)(λ::Real, F::PosSimplexFace) = λ*F.S
Base.length(F::PosSimplexFace) = F.n
rank(F::PosSimplexFace) = F.k
vec(F::PosSimplexFace) = F.S
face_name(F::PosSimplexFace) = "Face of positive simplex"
face_parameters(F::PosSimplexFace) = "rank = $(rank(F)); n = $(length(F))"
