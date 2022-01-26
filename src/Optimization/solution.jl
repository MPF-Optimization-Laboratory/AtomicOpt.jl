"""
Output structure 
"""
mutable struct Solution{T1<:AbstractFace, T2<:Vector{Float64}, T3<:Float64}
    F::T1
    Fnew::T1
    c::T2
    cnew::T2
    feas::T3
    feasnew::T3
    r::T2
    function Solution(M::AbstractLinearOp, b::Vector{Float64}, A::AbstractAtomicSet)
        z = M'*b
        F = face(A, z)
        Fnew = deepcopy(F)
        c = zeros( size(M*F, 2))
        cnew = deepcopy(c)
        feas = norm(b)^2/2
        feasnew = deepcopy(feas)
        r = deepcopy(b)
        new{AbstractFace, Vector{Float64},Float64}(F, Fnew, c, cnew, feas, feasnew, r)
    end
end

function update!(sol::Solution)
    sol.F = deepcopy(sol.Fnew)
    sol.c = deepcopy(sol.cnew)
    sol.feas = sol.feasnew
end

function constructPrimal(sol::Solution)
    if typeof(sol.F) <: SumFace
        rs = [rank(face) for face in sol.F.faces]
        cs = BlockArray(sol.c, rs)
        xs = [sol.F.faces[i]*cs[Block(i)] for i in 1:length(sol.F.faces)]
        return xs
    else
        return sol.F * sol.c
    end
end
