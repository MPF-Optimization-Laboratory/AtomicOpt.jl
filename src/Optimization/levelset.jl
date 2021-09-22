# struct LevelSetMethod{matT}
#     oracle::DualCGIterable
#     M::matT
#     b::Vector{Float64}
#     atomicSet::AbstractAtomicSet
#     α::Float64
#     feaTol::Float64
#     optTol::Float64
#     gapTol::Float64
#     maxIts::Integer
#     function LevelSetMethod(M::matT, b, atomicSet, α,
#                             feaTol, optTol, gapTol,
#                             maxIts) where matT
#         dCGiterator = DualCGIterable(M, b, atomicSet)
#         new{matT}(dCGiterator, M, b, atomicSet, α,
#                   feaTol, optTol, gapTol, maxIts)
#     end
# end

struct LevelSetMethod{matT,atomT<:AbstractAtomicSet}
    M::matT
    b::Vector{Float64}
    atomicSet::atomT
    α::Float64
    feaTol::Float64
    optTol::Float64
    gapTol::Float64
    maxIts::Int64
    function LevelSetMethod(M::matT, b, atomicSet::atomT,
                            α, feaTol, optTol, gapTol,
                            maxIts) where {matT, atomT}
        new{matT,atomT}(M, b, atomicSet, α,
                        feaTol, optTol, gapTol, maxIts)
    end
end

"""
Return an exit flag for the level-set oracle.
"""
function checkExit(p::LevelSetMethod, gap, ℓ, u, k, maxIts)
    u < p.α + p.feaTol && return :feasible     # |Mx-b|^2/2 < α is nearly satisfied
    gap ≤ p.feaTol/2 && return :optimal        # current lasso problem is fully optimal
    ℓ > p.α + p.feaTol && return :suboptimal   # ℓ is large enough to update τ
    k ≥ maxIts && return :iterations           # out of iterations
    return :noerror
end


"""
Dual conditional gradient method for the problem

    minimize 1/2||Mx-b||^2  subj to  gauge(x|A)≤τ

Includes various stopping conditions needed for
the level-set methods.
"""
function oracle!(p::LevelSetMethod, oracle, τ, maxIts)
    setRadius!(oracle, τ)
    k = 0; u = ℓ = 0.0
    exitFlag = :noerror
    for (gap, r) in oracle
        k = k + 1
        u = norm(r)^2/2
        ℓ = max(p.α, u - gap)
        exitFlag = checkExit(p, gap, ℓ, u, k, maxIts)
        exitFlag == :noerror || break
    end
    z = getNegGradient(oracle)
    s = support(p.atomicSet, z)
    return u, ℓ, s, k, exitFlag
end


"""
Level-set method for the problem

    min gauge(x | A) subj to Mx=b.
"""
function level_set(M, b::Vector, atomicSet::AbstractAtomicSet;
                   α::Float64 = 0.0,
                   tol::Float64 = 1e-12,
                   gapTol::Float64 = 1+norm(b),
                   maxIts::Int = size(M,2),
                   pr::Bool = true,
                   logger::Int = 2
                   )
    t0 = time()
    feaTol = tol*(1+norm(b))
    optTol = feaTol
    dcg = DualCGIterable(M, b, atomicSet)
    lvl = LevelSetMethod(M, b, atomicSet, α, feaTol, optTol, gapTol, maxIts)
    (logger == 2) && logger_head(lvl)
    (logger == 1) && logger_head_small(lvl)

    # Check whether b is zero
    if norm(b) < tol
        m, n = size(M)
        return zeros(n), 0.0
    end

    # Solve the subproblem for τ = 0
    τ = 0.0
    u = ℓ = norm(b)^2/2
    z = getNegGradient(dcg)
    s = support(atomicSet, z)

    # ----------------------------------------------------------------
    # dual CG iterations
    # ----------------------------------------------------------------
    k = 0; totIts = 0; exitFlag = nothing
    feas = NaN; feasbest = Inf
    x = c = cbest = F = Fbest = nothing
    while true
        k = k + 1

        # --------------------------------------------------------------
        # Update level-set radius.
        # --------------------------------------------------------------
        # Adjust lower bound and slope for root-finding on unsquared value.
        # r = lvl.oracle.r
        # s = s/norm(r)
        # ℓ = dot(b,r)/norm(r) - τ*s
        τ = τ + (ℓ-α)/s
        exitFlag == :iterations && break
        exitFlag == :feasible && break
        
        # --------------------------------------------------------------
        # Call oracle to obtain lower minorant defined by (ℓ,s).
        # --------------------------------------------------------------
        u, ℓ, s, minorIts, exitFlag = oracle!(lvl, dcg, τ, maxIts-totIts)

        # --------------------------------------------------------------
        # Second-order correction. (primal recovery)
        # --------------------------------------------------------------
        if pr && u - α ≤ gapTol
            c, F, feas = primalrecover(dcg, α)
            if feas ≤ feaTol
                exitFlag = :feasible
            # elseif feas < u && all(>=(0), c)
            #     replaceIterates!(lvl.oracle, M, F, c)
            end
        end

        # --------------------------------------------------------------
        # Keep track of the best recovery
        # --------------------------------------------------------------
        if feas < feasbest + 0.5
            feasbest = feas
            cbest = c
            Fbest = F
        end

        # --------------------------------------------------------------
        # Logging and bookkeeping.
        # --------------------------------------------------------------
        (logger == 2) && logger_level(lvl, τ, k, minorIts, ℓ, u, exitFlag, feas)
        (logger == 1) && (mod(k, 20) == 0) && logger_level_small(τ, k, ℓ, u, feas)
        totIts += minorIts
    end

    if cbest == nothing 
        cbest, Fbest, feasbest = primalrecover(dcg, α)
    end
    x = Fbest*cbest

    (logger == 2) && logger_foot(lvl, feasbest, totIts, u-ℓ, exitFlag)
    (logger == 1) && logger_foot_small(lvl, feasbest, totIts, u-ℓ, exitFlag)
    return x, τ
end

function logger_head(p::LevelSetMethod)
    m, n = size(p.M)
    println()
    println("  -------------------------------------------------------------------------")
    println("  Polar Level Set Method")
    println("  -------------------------------------------------------------------------")
    @printf("  %-22s %7d %7s %-20s %8.2e\n","number of variables",n,"","optimality",p.optTol)
    @printf("  %-22s %7d %7s %-20s %8.2e\n","number of constraints",m,"","feasibility",p.feaTol)
    @printf("  %-22s %7d %7s %-20s %8.2e\n","max duCG iterations",p.maxIts,"","α",p.α)
    println("  -------------------------------------------------------------------------")
    @printf("  %5s   %8s   %8s   %8s   %8s   %8s  %15s  %10s\n",
            "Major","Minor","u-α","ℓ-α","Gap","Tau","infeas(recover)","Subproblem")
end
function logger_foot(p::LevelSetMethod, normr, totIts, gap, exitFlag)
    scale = 1+norm(p.b)
    println("  -------------------------------------------------------------------------")
    @printf("  %-22s %8.1e %7s %-22s %8.2e\n","residual (abs)",normr,"","optimality gap (abs)",gap)
    @printf("  %-22s %8.1e %7s %-22s %8.2e\n","residual (rel)",normr/scale,"","optimality gap (rel)",gap/scale)
    @printf("  %-22s %8d %7s %-22s %8s\n","total iterations",totIts,"","","")
    println("  -------------------------------------------------------------------------")
end
function logger_level(p::LevelSetMethod, τ, k, minorItns, ℓ, u, exitFlag, feas)
    gap = u - ℓ
    # optimal = log10(dot(p.b,p.oracle.r)/norm(p.oracle.z,Inf))
    @printf("  %5d   %8d   %8.2e   %8.2e   %8.2e   %8.2e   %12.2e   %s\n",
            k, minorItns, u - p.α, ℓ-p.α, gap, τ, feas, exitFlag)
end

function logger_head_small(p::LevelSetMethod)
    m, n = size(p.M)
    println()
    println("  -------------------------------------------------------------------------")
    println("  Polar Level Set Method")
    println("  -------------------------------------------------------------------------")
    @printf("  %-22s %7d %7s %-20s %8.2e\n","number of variables",n,"","optimality",p.optTol)
    @printf("  %-22s %7d %7s %-20s %8.2e\n","number of constraints",m,"","feasibility",p.feaTol)
    @printf("  %-22s %7d %7s %-20s %8.2e\n","max duCG iterations",p.maxIts,"","α",p.α)
    println("  -------------------------------------------------------------------------")
    @printf("  %5s   %8s   %8s  %8s\n",
            "Major","Gap","Tau","Feasible")
end
function logger_foot_small(p::LevelSetMethod, normr, totIts, gap, exitFlag)
    scale = 1+norm(p.b)
    println("  -------------------------------------------------------------------------")
    # @printf("  %-22s %8.1e %7s %-22s %8.2e\n","residual (abs)",normr,"","optimality gap (abs)",gap)
    @printf("  %-22s %8.1e %7s %-22s %8.2e\n","residual (rel)",normr/scale,"","optimality gap (rel)",gap/scale)
    println("  -------------------------------------------------------------------------")
end
function logger_level_small(τ, k, ℓ, u, feas)
    gap = u - ℓ
    # optimal = log10(dot(p.b,p.oracle.r)/norm(p.oracle.z,Inf))
    @printf("  %5d   %8.2e   %8.2e   %8.2e \n",
            k, gap, τ, feas)
end
