"""
Output structure 
"""
mutable struct Solution{T1<:AbstractFace, T2<:Vector{Float64}, T3<:Float64}
    F::T1
    c::T2
    feas::T3
    function Solution(M::AbstractLinearOp, b::Vector{Float64}, A::AbstractAtomicSet)
        z = M'*b
        F = face(A, z)
        c = zeros( size(M*F, 2))
        feas = norm(b)^2/2
        new{AbstractFace, Vector{Float64},Float64}(F, c, feas)
    end
end

function constructPrimal(sol::Solution)
    return sol.F * sol.c
end
