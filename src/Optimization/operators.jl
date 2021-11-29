# some useful operators
struct MaskOP <: AbstractOperator
    I::Vector{Int64}
    J::Vector{Int64}
    M::SparseMatrixCSC{Float64, Int64}
    function MaskOP(M::SparseMatrixCSC{Float64, Int64})
        I, J, ~ = findnz(M)
        new(I, J, M)
    end
end

size(Mop::MaskOP) = size(Mop.M)


struct TMaskOP <: AbstractOperator
    colptr::Vector{Int64}
    rowval::Vector{Int64}
    M::SparseMatrixCSC{Float64, Int64}
    function TMaskOP(M::SparseMatrixCSC{Float64, Int64})
        colptr = M.colptr
        rowval = M.rowval
        new(colptr, rowval, M)
    end
end

Base.adjoint(Mop::MaskOP) = TMaskOP(Mop.M)
function Base.:(*)(Mop::MaskOP, a::NucBallAtom)
    u, v = a.u, a.v
    I, J = Mop.I, Mop.J
    V = []
    for k=1:length(I)
        i, j = I[k], J[k]
        append!(V, u[i]*v[j])
    end
    A = sparse(I, J, V)
    return A.nzval
end

function Base.:(*)(TMop::TMaskOP, y::Vector{Float64})
    m, n = size(TMop.M)
    colptr, rowval = TMop.colptr, TMop.rowval
    return SparseMatrixCSC{Float64, Int64}(m, n, colptr, rowval, y)
end
