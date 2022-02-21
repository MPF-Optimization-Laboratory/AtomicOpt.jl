########################################################################
# Conditional gradient method
########################################################################
"""
Structure to hold the iterates of the accelerated dual conditional gradient method.

AccDualCGIterable(M, b, A; optTol=√eps)

Construct a dual conditional gradient iterator for the problem

    minimize  ½‖Mx-b‖²  subject to gauge(A|x) ≤ 1.   

The absolute tolerance `optTol` is used to determine the stopping condition:

    gap ≤ optTol(1+‖b‖₂)

"""
mutable struct AccDualCGIterable{T1<:Float64, T2<:Vector{T1}, T3<:AbstractArray{T1}, T4<:AbstractLinearOp, T5<:AbstractAtomicSet}
    M::T4           # linear operator
    b::T2           # observation
    A::T5           # atomic set
    r::T2           # residual
    z::T3           # negative gradient
    Ma::T3          # image of the atom
    Mx::T3          # image of primal variable
    My::T3          # image of primal variable 2
    τ::T1           # current scale of atomic set
    optTol::T1      # tolerance
    function AccDualCGIterable(M::AbstractLinearOp, b::Vector{Float64}, A::AbstractAtomicSet, τ::Float64, optTol = sqrt(eps(1.0)))
        r = copy(b)
        z = M'*r
        Ma = M*expose(A, z)
        Mx = copy(Ma); fill!(Mx, 0.0)
        My = copy(Mx)
        new{Float64, Vector{Float64}, AbstractArray{Float64}, AbstractLinearOp, AbstractAtomicSet}(M, b, A, r, z, Ma, Mx, My, τ, optTol)
    end
end

function oracleExit(p::AccDualCGIterable, gap::Float64)
    return gap < p.optTol
end

"""
Defines one complete iteration of the dual conditional gradient method.
"""
function iterate(p::AccDualCGIterable, k::Int=0)
    # set δ
    δ = 2 / (k+3)
    # compute My
    p.My .= (1-δ)*p.Mx + δ*p.Ma
    # compute z
    p.z .*= (1-δ)
    r = p.M'*(p.b - p.My)
    p.z .+= δ*r
    # Compute  M*a  where  a = expose(τA, z)
    a = expose(p.A, p.z)
    p.Ma .= p.M*a
    rmul!(p.Ma, p.τ)
    # check exit
    gap = dot(p.Ma - p.Mx, p.r)
    oracleExit(p, gap) && return (gap, p.r), k+1
    # compute Mx
    p.Mx .*= (1-δ)
    p.Mx .+= δ*p.Ma
    # compute r
    p.r .= p.b - p.Mx
    
    return (gap, p.r), k+1
end


"""
Return an exit flag for the dual conditional gradient algorithm.
"""
function checkExitCg(gap::Float64, u::Float64, k::Int64, α::Float64, feaTol::Float64, maxIts::Int64)
    u < α + feaTol && return :feasible
    gap ≤ feaTol/2 && return :optimal                   # current lasso problem is fully optimal
    k ≥ maxIts && return :iterations                    # out of iterations
    return :noerror                                     # otherwise
end

"""
Dual conditional gradient method for the problem

    min 1/2||Mx-b||^2 subj to gauge(x | A) ≤ τ
"""
function conditional_graident(M::AbstractLinearOp, b::Vector{Float64}, A::AbstractAtomicSet, τ::Float64;
                   α::Float64 = 0.0,
                   tol::Float64 = 1e-12,
                   gapTol::Float64 = 1+norm(b),
                   maxIts::Int = size(M,2),
                   pr::Bool = true,
                   logger::Bool = true
                   )

    feaTol = tol*(1+norm(b))
    # Structure to hold the iterates
    dcg = DualCGIterable(M, b, A); setRadius!(dcg, τ) 
    # Structure to hold the solution
    sol = Solution(M, b, A)

    # head logging
    m, n = size(M)
    logger && logger_head_cg(m, n, feaTol, α, maxIts)

    # Check whether b is zero
    if norm(b) < tol
        return sol
    end

    # ----------------------------------------------------------------
    # min loop
    # ----------------------------------------------------------------
    k = 0
    exitFlag = :noerror
    for (gap, r) in dcg

        exitFlag == :noerror || break
        k = k + 1
        u = norm(r)^2/2

        exitFlag = checkExitCg(gap, u, k, α, feaTol, maxIts)
        
        # --------------------------------------------------------------
        # Primal recovery.
        # --------------------------------------------------------------
        if pr 
            exitFlag = primalrecover!(dcg, sol, α, feaTol, exitFlag)
        end

        # --------------------------------------------------------------
        # Logging and bookkeeping.
        # --------------------------------------------------------------
        logger && logger_level_cg(α, k, u, gap, exitFlag, sol.feas)
    end


    primalrecover!(dcg, sol, α, feaTol, exitFlag)

    
    # foot logging
    logger&& logger_foot_cg(b, sol.feas, k)

    return sol
end

function logger_head_cg(m, n, feaTol, α, maxIts)
    println()
    println("  -------------------------------------------------------------------------")
    println("  Dual Conditional Gradient Method")
    println("  -------------------------------------------------------------------------")
    @printf("  %-22s %7d %7s %-20s %7d\n","number of variables",n,"","number of constraints",m)
    @printf("  %-22s %7.2e %7s %-20s %7.2e\n","feasibility tolerance",feaTol,"","α",α)
    @printf("  %-22s %7d \n","max iterations",maxIts)
    println("  -------------------------------------------------------------------------")
    @printf("  %8s   %8s   %8s   %8s   %10s\n",
            "iteration","u-α","gap","infeas-α","exitFlag")
end

function logger_foot_cg(b, feas, totIts)
    scale = 1+norm(b)
    normr = sqrt(2*feas)
    println("  -------------------------------------------------------------------------")
    @printf("  %-22s %8.1e \n","residual (abs)",normr)
    @printf("  %-22s %8.1e \n","residual (rel)",normr/scale)
    @printf("  %-22s %8d \n","total iterations",totIts)
    println("  -------------------------------------------------------------------------")
end

function logger_level_cg(α, k, u, gap, exitFlag, feas)
    @printf("  %8d   %8.2e   %8.2e   %8.2e   %s\n",
            k, u-α, gap, feas-α, exitFlag)
end


