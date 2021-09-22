"""
    ResOneBall(z, λ)

Atomic set defined by the face of 1-norm ball exposed by vector z
"""
struct ResOneBall <: AbstractAtomicSet
    z::Vector
    λ::Real
    function ResOneBall(z, λ=1.0)
        λ < 0 && error("λ must be nonnegative")
        new(z, λ)
    end
end

expose(A::ResOneBall, z; kwargs...) = expose!(A::ResOneBall, copy(z); kwargs...)

"""
    expose!(A::ResOneBall, z::Vector; tol=1e-12)

Obtain an atom in the face exposed by the vector `z`.
The vector `z` is overwritten. If `norm(z,Inf)<tol`, then
`z` is returned untouched.
"""
function expose!(A::ResOneBall, x::Vector; tol=1e-12)
    n = length(A.z)
    a = expose(OneBall(n), A.z)
    a = sign.(a)
    ax = a.*x
    xmax = maximum(ax)
    if xmax < tol
        return x
    end
    nnz = 0
    for (i, val) in enumerate(ax)
        if abs(xmax - val) < tol*abs(val)
            x[i] = sign(x[i])
            nnz += 1
        else
            x[i] = zero(eltype(x))
        end
    end
    return x ./= nnz
end

support(A::ResOneBall, x) =  dot(expose(A, x), x)*A.λ

Base.length(A::ResOneBall) = length(A.z)
atom_name(A::ResOneBall) = "Face of 1-norm ball exposed by z"
atom_description(A::ResOneBall) = "Face of 1-norm ball exposed by z"
atom_parameters(A::ResOneBall) = "n = $(length(A))"

