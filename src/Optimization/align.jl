import Base: iterate
import Base.Iterators.take

"""
Structure to hold the iterates of the dual conditional gradient
method.
"""
# mutable struct DualCGIterable{Tz, matT, LinOpT, T <: Float64}
#     M::LinOpT
#     b::Vector{T}
#     atomicSet::AbstractAtomicSet
#     r::Vector{T}     # residual
#     z::Tz            # negative gradient
#     Ma::matT         # image of the atom
#     Mx::matT         # image of primal variable
#     Δr::matT         # residual step
#     τ::T
#     optTol::T
# end

mutable struct DualCGIterable{Tz, matT, LinOpT, atomT<:AbstractAtomicSet}
    M::LinOpT
    b::Vector{Float64}
    atomicSet::atomT
    r::Vector{Float64} # residual
    z::Tz              # negative gradient
    Ma::matT           # image of the atom
    Mx::matT           # image of primal variable
    Δr::matT           # residual step
    τ::Float64
    optTol::Float64
end


"""
    DualCGIterable(M, b, atomicSet; optTol=√eps)

Construct a dual conditional gradient iterator for the problem

    minimize  ½‖Mx-b‖²  subject to gauge(A|x) ≤ 1.

The absolute tolerance `optTol` is used to determine the stopping condition:

    gap ≤ optTol(1+‖b‖₂)
"""
# function DualCGIterable(M,
#                         b::Vector{Float64},
#                         atomicSet::AbstractAtomicSet,
#                         optTol = sqrt(eps(1.0)))
#     r = copy(b)
#     z = M'*r
#     Ma = M*expose(atomicSet, z)
#     Mx = M*expose(atomicSet, zeros(size(z)))
#     Δr = Ma - Mx
#     τ = 0.0
#     DualCGIterable(M, b, atomicSet, r, z, Ma, Mx, Δr, τ, optTol)
# end

function DualCGIterable(M,
                        b::Vector{Float64},
                        atomicSet::AbstractAtomicSet,
                        optTol = sqrt(eps(1.0)))
    r = copy(b)
    z = M'*r
    Ma = M*expose(atomicSet, z)
    Mx = M*expose(atomicSet, zeros(size(z)))
    Δr = Ma - Mx
    τ = 0.0
    DualCGIterable(M, b, atomicSet, r, z, Ma, Mx, Δr, τ, optTol)
end

function oracleExit(p::DualCGIterable, gap)
    return gap < p.optTol
end


# """
# Defines one complete iteration of the dual conditional gradient method.

# TODO: in-place versions of expose and both linear ops (forward/adjoint).
# """
# function iterate(p::DualCGIterable, k::Int=0)
#     r, z, Ma, Mx, Δr, M, τ = p.r, p.z, p.Ma, p.Mx, p.Δr, p.M, p.τ
#     a = τ*expose(p.atomicSet, z)
#     Ma = M*a
#     Δr = Ma - Mx; Δrs = vec(sum(Δr, dims=2)) # looks argly, may find better way
#     gap = dot(Δrs, r)
#     oracleExit(p, gap) && return (gap, r), k+1
#     θ = linesearch(Δr, r)
#     Δr = updateΔr(Δr, θ); Δrs = vec(sum(Δr, dims=2))
#     @. r = r - Δrs
#     @. p.Mx = Mx + Δr
#     z .= M'*r
#     return (gap, r), k+1
# end

"""
Defines one complete iteration of the dual conditional gradient method.
TODO: in-place versions of expose and both linear ops (forward/adjoint).
"""
function iterate(p::DualCGIterable, k::Int=0)

    # Compute  M*a  where  a ∈ τA
    a = expose(p.atomicSet, p.z)
    p.Ma .= p.M*a
    rmul!(p.Ma, p.τ)

    # Compute search direction
    @. p.Δr = p.Ma - p.Mx

    # Compute gap and possible exit
    gap = sumdot(p.Δr, p.r)
    oracleExit(p, gap) && return (gap, p.r), k+1

    # Linesearch
    α = linesearch(p.Δr, p.r)

    # r ← r - Δr*α
    mul!(p.r, p.Δr, α, -1, 1)

    # Update each column: Mx ← Mx + Δr*Diagonal(α)
    broadcast!(*, p.Δr, p.Δr, α')
    BLAS.axpy!(1, p.Δr, p.Mx)

    # Update gradient: z ← M'r
    mul!(p.z, p.M', p.r)

    return (gap, p.r), k+1
end

"""
    sumdot(X::Matrix, y::Vector)
Compute the sum of the dotproducts of each column of `X`
with the vector `y`.
"""
function sumdot(X::Matrix{Float64}, y::Vector{Float64})
    n = size(X, 1)
    fdot = x -> BLAS.dot(n, x, 1, y, 1)
    return sum(fdot, eachcol(X))
end
sumdot(x::Vector, y::Vector) = BLAS.dot(length(x), x, 1, y, 1)

"""
    setRadius!(p::DualCGIterable, τ::Float64)

Increase to `τ` the radius of the current Lasso problem. This routine
reflects the increase in the radius on various quantities of the dual
CG method.

If the current radius 0, then this routine only sets the new parameter,
but doesn't rescale any of the dual iterates.
"""
function setRadius!(p::DualCGIterable, τ::Float64)
    τ ≥ p.τ || throw(DomainError((τ,p.τ), "new τ must increase"))
    if p.τ > 0
        rescaleIterates!(p, τ/p.τ)
    end
    p.τ = τ
end
function rescaleIterates!(p::DualCGIterable, s::Real)
    p.Mx .*= s
    Mxs = vec(sum(p.Mx, dims=2))
    p.r .= p.b .- Mxs
    p.z .= p.M'*p.r
end

function replaceIterates!(p::DualCGIterable, M::AbstractMatrix, F::AbstractFace, c::Vector)
    MF = M*F
    p.Mx .= MF*c
    p.Mx .*= p.τ/sum(c)
    Mxs = vec(sum(p.Mx, dims=2))
    p.r .= p.b .- Mxs
    p.z .= p.M'*p.r
end

getNegGradient(p::DualCGIterable) = p.z
getResidual(p::DualCGIterable) = p.r


"""
    linesearch(p::Vector, q::Vector)

Solve univariate least-squares problem

    minimize_{θ∈[0,1]} ||θp-q||₂,

which has the solution

    θ = min(1, dot(p,q)/dot(p,p))
"""
function linesearch(p::Vector, q::Vector)
    return min(1.0, dot(p,q)/dot(p,p))
end

"""
    linesearch(A::Matrix, b::Vector)

Solve the box-constrained least squares problem

    minimize_{x∈[0,1]ⁿ} ||Ax-b||₂
"""
function linesearch(A::Matrix, b::Vector)
    n = size(A, 2)
    bl = zeros(n)
    bu = ones(n)
    # x, ~, ~ = lsbox(A, b, bu, verbose=false)
    Q = Quadratic(A'*A, A'b)
    lbfgsq = LBFGSBQuad(Q, bl, bu, m=2)
    x = boxquad(lbfgsq)
    return x
end


# """
#     update Δr with θ
# """
# function updateΔr(Δr::Vector, θ::Real)
#     return θ*Δr
# end

# function updateΔr(Δr::Matrix, θ::Vector)
#     return Δr*Diagonal(θ)
# end

"""
    primal recover (need inplace version)
"""
function primalrecover(p::DualCGIterable, α::Real)
    s = support(p.atomicSet, p.z)
    F = face(p.atomicSet, p.z/s)
    ϵ = getResidual(p); ϵ .*= sqrt(2*α)/norm(ϵ)
    # println("start primal recovery")
    c, r = face_project(p.M, F, p.b - ϵ)
    feas = norm(r + ϵ)^2/2 - α
    return c, F, feas
end



