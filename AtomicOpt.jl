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
using Arpack
using LBFGSB

import Base: show, vec
import LinearAlgebra: dot, ×, cross, rank
# import LinearMaps: ⊗

export AbstractAtomicSet, AbstractAtom, AbstractFace
export MappedAtomicSet, MappedAtom, MappedFace
export ScaledAtomicSet, ScaledAtom, ScaledFace
export SumAtomicSet, SumAtom, SumFace
export CrossProductSet, CrossProductAtom, CrossProductFace
export OneBall, OneBallAtom, OneBallFace
export NucBall, NucBallAtom, NucBallFace
export BlkNucBall, BlkNucBallAtom, BlkNucBallFace
export PosSimplex, PosSimplexAtom, PosSimplexFace
export TwoBall, TwoBallAtom, TwoBallFace
# export TraceBall, TraceBallAtom
export face, gauge, rank, support, expose, expose!
export face_project
export ×, cross
export level_set

"Abstract atomic set."
abstract type AbstractAtomicSet end

"Abstract atoms."
abstract type AbstractAtom end

"Abstract face."
abstract type AbstractFace end

"Abstract operators."
abstract type AbstractOperator end

"Abstract Linear Operator"
AbstractLinearOp = Union{LinearMap, AbstractMatrix}

include("src/BasicSets/OneBall.jl")
include("src/BasicSets/NucBall.jl")
include("src/BasicSets/BlkNucBall.jl")
include("src/BasicSets/TraceBall.jl")
include("src/BasicSets/PosSimplex.jl")
include("src/BasicSets/TwoBall.jl")
include("src/SetOperations/mapped.jl")
include("src/SetOperations/scaled.jl")
include("src/SetOperations/sum.jl")
include("src/SetOperations/crossproduct.jl")
include("src/SetOperations/facialprojection.jl")
include("src/SetOperations/utils.jl")
include("src/Optimization/align.jl")
include("src/Optimization/levelset.jl")
include("src/Optimization/boxls.jl")

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
