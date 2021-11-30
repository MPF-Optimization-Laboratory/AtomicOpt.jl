# some useful operators
struct MaskOP <: AbstractOperator
    m::Int64
    n::Int64
    nnz::Int64
    I::Vector{Int64}
    J::Vector{Int64}
    M::SparseMatrixCSC{Float64, Int64}
    function MaskOP(M::SparseMatrixCSC{Float64, Int64})
        m, n = size(M)
        I, J, ~ = findnz(M)
        nnz = length(I)
        new(m, n, nnz, I, J, M)
    end
end

size(Mop::MaskOP) = Mop.nnz, Mop.m * Mop.n


struct TMaskOP <: AbstractOperator
    m::Int64
    n::Int64
    nnz::Int64
    colptr::Vector{Int64}
    rowval::Vector{Int64}
    M::SparseMatrixCSC{Float64, Int64}
    function TMaskOP(M::SparseMatrixCSC{Float64, Int64})
        m, n = size(M)
        colptr = M.colptr
        rowval = M.rowval
        nnz = length(rowval)
        new(m, n, nnz, colptr, rowval, M)
    end
end

size(TMop::TMaskOP) = TMop.m * TMop.n, TMop.nnz

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

function Base.:(*)(Mop::MaskOP, X::Tuple{Matrix{Float64}, Matrix{Float64}})
    U, V = X[1], X[2]
    I, J = Mop.I, Mop.J
    Val = []
    for k=1:length(I)
        i, j = I[k], J[k]
        append!(Val, U[i,:]'*V[j,:])
    end
    A = sparse(I, J, Val)
    return A.nzval
end

function Base.:(*)(TMop::TMaskOP, y::Vector{Float64})
    m, n = TMop.m, TMop.n
    colptr, rowval = TMop.colptr, TMop.rowval
    return SparseMatrixCSC{Float64, Int64}(m, n, colptr, rowval, y)
end

function Base.:(*)(M::MaskOP, F::NucBallFace) 
    m, n, r = F.m, F.n, F.r
    U, V = F.U, F.V
    f = p->M * (U*reshape(p, r, r), V)
    fc = q->vec(F.U'*(M'*q)*V)
    return LinearMap(f, fc, nnz(M.M), r*r)
end
