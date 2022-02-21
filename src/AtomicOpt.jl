module AtomicOpt

using LinearAlgebra
using LinearMaps
using Printf
using Random
using SparseArrays
using Arpack
using IterativeSolvers
using Distributed
using FFTW
using LBFGSB
using BlockArrays

import Base: show, vec, size
import LinearAlgebra: dot, rank

export AbstractAtomicSet, AbstractAtom, AbstractFace
export MappedAtomicSet, MappedAtom, MappedFace
export SumAtomicSet, SumAtom, SumFace
export OneBall, OneBallAtom, OneBallFace
export NucBall, NucBallAtom, NucBallFace
export BlkNucBall, BlkNucBallAtom, BlkNucBallFace
# export PosSimplex, PosSimplexAtom, PosSimplexFace
# export TwoBall, TwoBallAtom, TwoBallFace
export TraceBall, TraceBallAtom, TraceBallFace
export face, face!, gauge, rank, support, expose, expose!
export face_project!
export level_set, conditional_graident, coordinate_descent
export Solution, constructPrimal
export MaskOP, TMaskOP
export dual_obj_gap

"Abstract atomic set."
abstract type AbstractAtomicSet end

"Abstract atoms."
abstract type AbstractAtom end

"Abstract face."
abstract type AbstractFace end

"Abstract operators."
abstract type AbstractOperator end

"Abstract Linear Operator"
LinearOp = Union{LinearMap, AbstractMatrix}
AbstractLinearOp = Union{LinearOp, AbstractOperator}

include("./BasicSets/OneBall.jl")
include("./BasicSets/NucBall.jl")
include("./BasicSets/BlkNucBall.jl")
include("./BasicSets/TraceBall.jl")
# include("src/BasicSets/PosSimplex.jl")
# include("src/BasicSets/TwoBall.jl")
include("./SetOperations/mapped.jl")
include("./SetOperations/sum.jl")
include("./SetOperations/operators.jl")
include("./SetOperations/facialprojection.jl")
include("./SetOperations/utils.jl")
include("./Optimization/solution.jl")
include("./Optimization/align.jl")
include("./Optimization/levelset.jl")
include("./Optimization/boxls.jl")
include("./Optimization/conditionalGradient.jl")
include("./Optimization/liftedCoordinateDescent.jl")



dot(a1::AbstractAtom, a2::Vector) = dot(vec(a1), a2)
dot(a1::Vector, a2::AbstractAtom) = dot(vec(a2), a1)
dot(a1::AbstractAtom, a2::AbstractAtom) = dot(vec(a1), vec(a2))

function Base.show(io::IO, A::AbstractAtomicSet)
    println(io, "atomic set  : ", atom_name(A))
    println(io, "description : ", atom_description(A))
    print(  io, "parameters  : ", atom_parameters(A))
end

function Base.show(io::IO, F::AbstractFace)
    println(io, "face of atomic set  : ", face_name(F))
    print(  io, "parameters          : ", face_parameters(F))
end

end # module

