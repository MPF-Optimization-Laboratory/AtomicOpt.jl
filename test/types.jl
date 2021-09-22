# Tests for checking type stability.
using AtomicSets

function demixing()
    As = [OneBall(2), OneBall(2)]
    A = reduce(cross, As)
end

@code_warntype demixing()


abstract type Atom end
struct Ni <: Atom
    w::Int64
end
struct Cd <: Atom
    w::Int64
end
struct CrossProduct{T<:Tuple{Vararg{Atom}}} <: Atom
    a::T
    function CrossProduct(a::T) where T
        return new{T}(a)
    end
end
function cross(A::Atom, B::Atom)
    return CrossProduct(tuple(A,B))
end
function cross(A::CrossProduct, B::Atom)
    return CrossProduct(tuple(A.a..., B))
end
function cross(A::Atom, B::CrossProduct)
    return CrossProduct(tuple(A, B.a...))
end
function cross(A::CrossProduct, B::CrossProduct)
    return CrossProduct(tuple(A.a..., B.a...))
end

function testcross()
    n, c = Ni(1), Cd(2)
    nc = cross(n, c)
    nc2 = reduce(cross, (n, c))
end

# @code_warntype testcross()
