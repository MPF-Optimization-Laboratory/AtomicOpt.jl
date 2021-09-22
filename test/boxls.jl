using AtomicOpt
using LinearAlgebra
using Random
using Test

import AtomicOpt: LBFGSBQuad, Quadratic, boxquad, grad

"""Project into the box [bl, bu]."""
proj(x, bl, bu) = max.(min.(x, bu), bl)

@testset "Box-constrained LS via projected Newton" begin
    m, n = 10, 3
    A = randn(m, n); x0 = randn(n); b = A*x0
    bl = zeros(n); bu = ones(n)
    x = randn(n)

    obj(x) = 0.5*norm(A*x)^2 - dot(A'b,x)

    Q = Quadratic(A'*A, A'b)
    @test obj(x) ≈ Q(x)

end

@testset "Box-constrained LS via LBFGS-B" begin
    m, n = 100, 30
    A = randn(m, n); x0 = randn(n); b = A*x0
    bl = rand(n) .- 0.5; bu = rand(n) .+ 0.5

    Q = Quadratic(A'*A, A'b)
    lbfgsq = LBFGSBQuad(Q, bl, bu)
    x = boxquad(lbfgsq, factr=1.0)
    @test norm( x - proj(x-grad(Q,x), bl, bu)) ≤ 1e-6*norm(x)
end
