import Base: iterate
import Base.Iterators.take

"""
Structure to hold the iterates of the dual conditional gradient method.

DualCGIterable(M, b, A; optTol=√eps)

Construct a dual conditional gradient iterator for the problem

    minimize  ½‖Mx-b‖²  subject to gauge(A|x) ≤ 1.   

The absolute tolerance `optTol` is used to determine the stopping condition:

    gap ≤ optTol(1+‖b‖₂)

"""
mutable struct DualCGIterable{T1<:Float64, T2<:Vector{T1}, T3<:AbstractArray{T1}, T4<:AbstractLinearOp, T5<:AbstractAtomicSet, T6<:AbstractAtom}
    M::T4           # linear operator
    b::T2           # observation
    A::T5           # atomic set
    r::T2           # residual
    z::T3           # negative gradient
    a::T6           # exposed atom
    Ma::T3          # image of the atom
    Mx::T3          # image of primal variable
    Δr::T3          # residual step
    τ::T1           # current scale of atomic set
    optTol::T1      # tolerance
    function DualCGIterable(M::AbstractLinearOp, b::Vector{Float64}, A::AbstractAtomicSet, optTol = sqrt(eps(1.0)))
        r = copy(b)
        z = M'*r
        a = expose(A, z)
        Ma = M*a
        Mx = copy(Ma); fill!(Mx, 0.0)
        Δr = Ma - Mx
        τ = 0.0
        new{Float64, Vector{Float64}, AbstractArray{Float64}, AbstractLinearOp, AbstractAtomicSet, AbstractAtom}(M, b, A, r, z, a, Ma, Mx, Δr, τ, optTol)
    end
end

function oracleExit(p::DualCGIterable, gap::Float64)
    return gap < p.optTol
end

"""
Defines one complete iteration of the dual conditional gradient method.
"""
function iterate(p::DualCGIterable, k::Int=0)

    # Compute  M*a  where  a = expose(τA, z)
    expose!(p.A, p.z, p.a)
    # p.Ma .= p.M*p.a
    mul!(p.Ma, p.M, p.a)
    rmul!(p.Ma, p.τ)

    # Compute search direction Δr = M(a - x)
    @. p.Δr = p.Ma - p.Mx
    
    # Compute gap and possible exit
    gap = sumdot(p.Δr, p.r)
    oracleExit(p, gap) && return (gap, p.r), k+1

    if iszero(p.Mx) 
        # Mx ← Ma
        p.Mx .= p.Ma
        # r ← b - Mx
        p.r .= p.b - sum(p.Mx, dims=2)[:]
    else
        # Linesearch
        α = linesearch(p.Δr, p.r)
        # r ← r - Δr*α
        mul!(p.r, p.Δr, α, -1, 1)
        # Mx ← Mx + Δr*Diagonal(α)
        broadcast!(*, p.Δr, p.Δr, α')
        BLAS.axpy!(1, p.Δr, p.Mx)
    end

    # Update gradient: z ← M'r
    mul!(p.z, p.M', p.r)

    return (gap, p.r), k+1
end

"""
    sumdot(X::Matrix, y::Vector)
Compute the sum of the dotproducts of each column of `X`
with the vector `y`.
"""
function sumdot(X::AbstractMatrix{Float64}, y::Vector{Float64})
    fdot = x -> dot(x, y)
    return sum(fdot, eachcol(X))
end

sumdot(x::Vector{Float64}, y::Vector{Float64}) = dot(x, y)


"""
    setRadius!(p::DualCGIterable, τ::Float64)

Increase to `τ` the radius of the current Lasso problem. This routine
reflects the increase in the radius on various quantities of the dual
CG method.

If the current radius 0, then this routine only sets the new parameter,
but doesn't rescale any of the dual iterates.
"""
function setRadius!(p::DualCGIterable, τ::Float64)
    if p.τ > 0
        rescaleIterates!(p, τ/p.τ)
    end
    p.τ = τ
end

function rescaleIterates!(p::DualCGIterable, s::Float64)
    p.Mx .*= s
    Mxs = vec(sum(p.Mx, dims=2))
    p.r .= p.b .- Mxs
    p.z .= p.M'*p.r
end

function replaceIterates!(p::DualCGIterable, M::AbstractMatrix{Float64}, F::AbstractFace, c::Vector{Float64})
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
    linesearch(Δr::Vector, r::Vector)

Solve univariate least-squares problem

    minimize_{θ∈[0,1]} ||θ⋅Δr-r||₂,

which has the solution

    θ = min(1, dot(Δr,Δr)/dot(Δr,Δr))
"""
function linesearch(Δr::Vector{Float64}, r::Vector{Float64})
    return min(1.0, dot(Δr,r)/dot(Δr,Δr))
end

"""
    linesearch(Δr::Matrix, r::Vector)

Solve the box-constrained least squares problem

    minimize_{θ∈[0,1]ⁿ} ||Δr⋅θ - r||₂
"""
function linesearch(Δr::AbstractMatrix{Float64}, r::Vector{Float64})
    n = size(Δr, 2)
    bl = zeros(n)
    bu = ones(n)
    Q = Quadratic(Δr'*Δr, Δr'r)
    lbfgsq = LBFGSBQuad(Q, bl, bu, m=2)
    θ = boxquad(lbfgsq)
    return θ
end

"""
    primal recover 
"""
function primalrecover!(p::DualCGIterable, sol::Solution, α::Float64, feaTol::Float64, exitFlag)
    flag = exitFlag
    s = support(p.A, p.z)
    # F = face(p.A, p.z/s)
    face!(p.A, p.z/s, sol.Fnew)
    ϵ = copy(getResidual(p)); ϵ .*= sqrt(2*α)/norm(ϵ)
    # c, r = face_project(p.M, F, p.b - ϵ)
    face_project!(p.M, sol.Fnew, p.b-ϵ, sol.cnew, sol.r)
    sol.feasnew = norm(sol.r + ϵ)^2/2 
    if sol.feasnew  ≤ sol.feas
        update!(sol)
    end
    if sol.feas - α ≤ feaTol
        flag = :feasible
    end
    return flag
end



