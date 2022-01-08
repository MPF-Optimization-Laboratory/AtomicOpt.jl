import Base: iterate
import Base.Iterators.take

"""
Structure to hold the iterates of the lifted coordinate descent method.

LiftedCCIterable(M, b, A, λ; optTol=√eps)

Construct a lifted coordinate descent iterator for the problem

    minimize  ½‖Mx-b‖² + λ⋅gauge(A|x)

The absolute tolerance `optTol` is used to determine the stopping condition.

"""
mutable struct LiftedCCIterable{T1<:Float64, T2<:Vector{T1}, T3<:AbstractArray{T1}, T4<:AbstractLinearOp, T5<:AbstractAtomicSet}
    M::T4           # linear operator
    b::T2           # observation
    A::T5           # atomic set
    r::T2           # residual
    z::T3           # negative gradient
    Ma::T3          # image of the atom
    Mx::T3          # image of primal variable
    λ::T1           # current scale of atomic set
    optTol::T1      # tolerance
    function LiftedCCIterable(M::AbstractLinearOp, b::Vector{Float64}, A::AbstractAtomicSet, λ::Float64, optTol = sqrt(eps(1.0)))
        r = copy(b)
        z = M'*r
        Ma = M*expose(A, z)
        Mx = copy(Ma); fill!(Mx, 0.0)
        new{Float64, Vector{Float64}, AbstractArray{Float64}, AbstractLinearOp, AbstractAtomicSet}(M, b, A, r, z, Ma, Mx, λ, optTol)
    end
end

"""
Defines one complete iteration of the lifted coordinate descent method.
"""
function iterate(p::LiftedCCIterable, k::Int=0)

    # Compute  M*a  where  a = expose(A, z)
    s = support(p.A, p.z)
    a = expose(p.A, p.z)
    p.Ma .= p.M*a

    # Compute g = λ - support(A|z) and possible exit
    g = p.λ - s

    # Compute step size
    δ = -g / (norm(p.Ma)^2)

    # update iterates
    p.Mx .+= δ*p.Ma
    p.r .= p.b - p.Mx
    p.z .= p.M' * p.r

    return (g, p.r), k+1
end

"""
Return an exit flag for the lifted coordinate descent algorithm.
"""
function checkExitCd(g::Float64, u::Float64, k::Int64, α::Float64, feaTol::Float64, maxIts::Int64)
    u < α + feaTol && return :feasible                  # feasible
    g > -sqrt(eps(1.0))/2 && return :optimal                    # optimal
    k ≥ maxIts && return :iterations                    # out of iterations
    return :noerror                                     # otherwise
end

"""
Lifted coordinate descent method for the problem

    min ½‖Mx-b‖² + λ⋅gauge(A|x)
"""
function coordinate_descent(M::AbstractLinearOp, b::Vector{Float64}, A::AbstractAtomicSet, λ::Float64;
                   α::Float64 = 0.0,
                   tol::Float64 = 1e-12,
                   gapTol::Float64 = 1+norm(b),
                   maxIts::Int = size(M,2),
                   pr::Bool = true,
                   logger::Bool = true
                   )

    feaTol = tol*(1+norm(b))
    # Structure to hold the iterates
    lcc = LiftedCCIterable(M, b, A, λ) 
    # Structure to hold the solution
    sol = Solution(M, b, A)

    # head logging
    m, n = size(M)
    logger && logger_head_cd(m, n, feaTol, α, maxIts)

    # Check whether b is zero
    if norm(b) < tol
        return sol
    end

    # ----------------------------------------------------------------
    # min loop
    # ----------------------------------------------------------------
    k = 0
    exitFlag = :noerror
    for (g, r) in lcc

        exitFlag == :noerror || break
        k = k + 1
        u = norm(r)^2/2
        exitFlag = checkExitCd(g, u, k, α, feaTol, maxIts)
        
        # --------------------------------------------------------------
        # Primal recovery.
        # --------------------------------------------------------------
        if pr 
            exitFlag = primalrecover!(lcc, sol, α, feaTol, exitFlag)
        end

        # --------------------------------------------------------------
        # Logging and bookkeeping.
        # --------------------------------------------------------------
        logger && logger_level_cd(α, k, u, g, exitFlag, sol.feas)
    end

    primalrecover!(lcc, sol, α, feaTol, exitFlag)
    
    # foot logging
    logger&& logger_foot_cd(b, sol.feas, k)

    return sol
end

function logger_head_cd(m, n, feaTol, α, maxIts)
    println()
    println("  -------------------------------------------------------------------------")
    println("  Lifted Coordinate Descent Method")
    println("  -------------------------------------------------------------------------")
    @printf("  %-22s %7d %7s %-20s %7d\n","number of variables",n,"","number of constraints",m)
    @printf("  %-22s %7.2e %7s %-20s %7.2e\n","feasibility tolerance",feaTol,"","α",α)
    @printf("  %-22s %7d \n","max iterations",maxIts)
    println("  -------------------------------------------------------------------------")
    @printf("  %8s   %8s   %8s   %8s   %10s\n",
            "iteration","u-α","g","infeas-α","exitFlag")
end

function logger_foot_cd(b, feas, totIts)
    scale = 1+norm(b)
    normr = sqrt(2*feas)
    println("  -------------------------------------------------------------------------")
    @printf("  %-22s %8.1e \n","residual (abs)",normr)
    @printf("  %-22s %8.1e \n","residual (rel)",normr/scale)
    @printf("  %-22s %8d \n","total iterations",totIts)
    println("  -------------------------------------------------------------------------")
end

function logger_level_cd(α, k, u, g, exitFlag, feas)
    @printf("  %8d   %8.2e   %8.2e   %8.2e   %s\n",
            k, u-α, g, feas-α, exitFlag)
end

"""
    primal recover 
"""
function primalrecover!(p::LiftedCCIterable, sol::Solution, α::Float64, feaTol::Float64, exitFlag)
    flag = exitFlag
    s = support(p.A, p.z)
    F = face(p.A, p.z/s)
    ϵ = copy(p.r); ϵ .*= sqrt(2*α)/norm(ϵ)
    c, r = face_project(p.M, F, p.b - ϵ)
    feas = norm(r + ϵ)^2/2 
    @show feas
    if feas ≤ sol.feas
        sol.F = F
        sol.c = c
        sol.feas = feas
    end
    if feas - α ≤ feaTol
        flag = :feasible
    end
    return flag
end