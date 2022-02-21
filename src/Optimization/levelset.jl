########################################################################
# Level set method
########################################################################
"""
Return an exit flag for the level-set oracle.
"""
function checkExitLvl(gap::Float64, ℓ::Float64, u::Float64, k::Int64, α::Float64, feaTol::Float64, maxIts::Int64, rule::String)
    if rule == "newton"
        # u < α + feaTol && return :feasible              # |Mx-b|^2/2 < α is nearly satisfied
        ℓ > α + feaTol && return :suboptimal            # ℓ is large enough to update τ
    elseif rule == "bisection"
        u < α && return :suboptimal_large               # τ too large
        ℓ > α && return :suboptimal_small               # τ too small
    end
    gap ≤ feaTol/2 && return :optimal                   # current lasso problem is fully optimal
    k ≥ maxIts && return :iterations                    # out of iterations
    return :noerror                                     # otherwise
end


"""
Dual conditional gradient method for the problem

    minimize 1/2||Mx-b||^2  subj to  gauge(x|A)≤τ

Includes various stopping conditions needed for
the level-set methods.
"""
function solveSubproblem!(dcg::DualCGIterable, τ::Float64, α::Float64, feaTol::Float64, maxIts::Int64, rule::String)
    setRadius!(dcg, τ)
    k = 0; u = 0.0; ℓ = 0.0
    exitFlag = :noerror
    for (gap, r) in dcg
        k = k + 1
        u = norm(r)^2/2
        ℓ = max(α, u - gap)
        exitFlag = checkExitLvl(gap, ℓ, u, k, α, feaTol, maxIts, rule)
        exitFlag == :noerror || break
    end
    s = support(dcg.A, dcg.z)
    return u, ℓ, s, k, exitFlag
end


"""
Level-set method for the problem

    min gauge(x | A) subj to 1/2||Mx-b||^2 ≤ α
"""
function level_set(M::AbstractLinearOp, b::Vector{Float64}, A::AbstractAtomicSet;
                   α::Float64 = 0.0,
                   tol::Float64 = 1e-12,
                   gapTol::Float64 = 1+norm(b),
                   maxIts::Int = size(M,2),
                   pr::Bool = true,
                   logger::Bool = true,
                   rule::String = "newton",
                   τmax::Float64 = 1.0,
                   callback::Function = identity
                   )

    feaTol = tol*(1+norm(b))
    # Structure to hold the iterates
    dcg = DualCGIterable(M, b, A) 
    # Structure to hold the solution
    sol = Solution(M, b, A)

    # head logging
    m, n = size(M)
    logger && logger_head_lvl(m, n, feaTol, α, maxIts)

    # Check whether b is zero
    if norm(b) < tol
        return sol
    end

    # Initialization at τ = 0
    τ = 0.0
    u = norm(b)^2/2; ℓ = norm(b)^2/2
    if rule == "newton"
        z = getNegGradient(dcg)
        s = support(A, z)
    elseif rule == "bisection"
        τmin = 0.0
    else
        println("invalid updating rule")
        return sol
    end

    # ----------------------------------------------------------------
    # min loop
    # ----------------------------------------------------------------
    k = 0; totIts = 0; exitFlag = :suboptimal_small
    while true
        k = k + 1

        # --------------------------------------------------------------
        # Update level-set radius.
        # --------------------------------------------------------------
        if rule == "newton"
            τ = τ + (ℓ-α)/s
        elseif rule == "bisection"
            if exitFlag == :suboptimal_small
                τmin = τ
            elseif exitFlag == :suboptimal_large
                τmax = τ
            end
            τ = (τmin + τmax)/2
            ρ = τmax - τmin
            @show ρ
        end

        # --------------------------------------------------------------
        # Check for exit.
        # --------------------------------------------------------------
        exitFlag == :iterations && break
        exitFlag == :feasible && break
        exitFlag == :optimal && break
        
        # --------------------------------------------------------------
        # Solve subproblem to obtain lower minorant defined by (ℓ,s).
        # --------------------------------------------------------------
        u, ℓ, s, minorIts, exitFlag = solveSubproblem!(dcg, τ, α, feaTol, maxIts-totIts, rule)

        # --------------------------------------------------------------
        # Primal recovery.
        # --------------------------------------------------------------
        # if pr && u - α ≤ gapTol
        if pr
            exitFlag = primalrecover!(dcg, sol, α, feaTol, exitFlag)
        end

        # --------------------------------------------------------------
        # Callback
        # --------------------------------------------------------------
        callback(dcg)

        # --------------------------------------------------------------
        # Logging and bookkeeping.
        # --------------------------------------------------------------
        logger && logger_level_lvl(α, τ, k, minorIts, ℓ, u, exitFlag, sol.feas)
        totIts += minorIts
    end


    primalrecover!(dcg, sol, α, feaTol, exitFlag)

    
    # foot logging
    logger&& logger_foot_lvl(b, sol.feas, totIts, u-ℓ)

    return sol
end

function logger_head_lvl(m, n, feaTol, α, maxIts)
    println()
    println("  -------------------------------------------------------------------------")
    println("  Polar Level Set Method")
    println("  -------------------------------------------------------------------------")
    @printf("  %-22s %7d %7s %-20s %7d\n","number of variables",n,"","number of constraints",m)
    @printf("  %-22s %7.2e %7s %-20s %7.2e\n","feasibility tolerance",feaTol,"","α",α)
    @printf("  %-22s %7d \n","max iterations",maxIts)
    println("  -------------------------------------------------------------------------")
    @printf("  %5s   %8s   %8s   %8s   %8s   %8s  %15s  %10s\n",
            "Major","Minor","u-α","ℓ-α","gap","τ","infeas-α","Subproblem")
end

function logger_foot_lvl(b, feas, totIts, gap)
    scale = 1+norm(b)
    normr = sqrt(2*feas)
    println("  -------------------------------------------------------------------------")
    @printf("  %-22s %8.1e %7s %-22s %8.2e\n","residual (abs)",normr,"","optimality gap (abs)",gap)
    @printf("  %-22s %8.1e %7s %-22s %8.2e\n","residual (rel)",normr/scale,"","optimality gap (rel)",gap/scale)
    @printf("  %-22s %8d %7s %-22s %8s\n","total iterations",totIts,"","","")
    println("  -------------------------------------------------------------------------")
end

function logger_level_lvl(α, τ, k, minorItns, ℓ, u, exitFlag, feas)
    gap = u - ℓ
    # optimal = log10(dot(p.b,p.oracle.r)/norm(p.oracle.z,Inf))
    @printf("  %5d   %8d   %8.2e   %8.2e   %8.2e   %8.2e   %12.2e   %s\n",
            k, minorItns, u-α, ℓ-α, gap, τ, feas-α, exitFlag)
end


function dual_obj_gap(p::DualCGIterable, τ::Float64, λ::Float64, α::Float64)
    β = τ
    s = support(p.A, p.z)
    y = deepcopy(p.r); y .*= 1 / s
    dobj = norm(y)^2/(2*β) + β*α - dot(y, p.b)
    return abs(dobj + τ)
end