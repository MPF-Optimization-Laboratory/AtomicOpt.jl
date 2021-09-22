using AtomicOpt
using Test
using LinearAlgebra
using Arpack
import Random: seed!, randperm
seed!(0)

@testset "OneBall" begin
    n = 3
    x = randn(n); z = randn(n)
    B = OneBall(n)
    @test length(B) == n
    @test gauge(B,x) == norm(x,1)
    @test support(B,z) == norm(z,Inf)
    @test dot(z, expose(B,z; tol=1e-12)) ≈ support(B,z)

    # in-place version of expose
    zc = copy(z)
    expose!(B,zc; tol=1e-12)
    @test dot(z,zc) ≈ support(B,z)

    # map an atom
    a = expose(B,z)
    M = rand(n,n)
    @test dot(M*a, x) ≈ dot(a, M'*x)

end

@testset "OneBallFace" begin
    n = 5
    A = OneBall(n); z = [-1.0,+1.0,0.0,0.0,-1.0]
    S = [-1. 0  0
         0. +1. 0
         0   0  0
         0   0  0
         0   0 -1.]
    F = face(A, z)
    c = randn(3)
    @test Matrix(I,n,n)*F*c == S*c
    @test vec(F)*c == S*c

    A = OneBall(n, maxrank=2)
    F = face(A, z)
    @test size(vec(F)) == (n,rank(A))

    # Facial projection
    M = randn(n,n)
    x0 = [2., 0., -3., 0., 0.]
    z  = [1., 0., -1., 0., 0.]
    b = M*x0
    F = face(A,z)
    c, r = face_project(M, F, b)
    @test c ≈ [abs(xi) for xi in x0 if !iszero(xi)]
    @test x0 ≈ F*c
    @test norm(r) ≈ 0.0 atol=1e-12
end

@testset "ScaledAtomicSet" begin
    n = 3;
    λ = rand(); x = randn(n); z = randn(n)
    B = OneBall(n)
    @test support(B,z) ≈ support(λ*B,z)/λ
    @test dot(x,λ*expose(B,z)) ≈ dot(x,expose(λ*B,z))
end

@testset "ScaledFace" begin
    n = 5;
    A = OneBall(n)
    z = randn(n)
    F = face(2*A, z)
    k = rank(F)
    c = randn(k)
    @test vec(F)*c == 2*face(A, z)*c
end

@testset "MappedAtomicSet" begin
    n = 3;
    x = randn(n); z = randn(n)
    W = rand(n,n)
    B = 2OneBall(n)
    @test length(W*B) == length(B)
    @test support(W*B, z) ≈ support(B, W'z)
    @test dot(x, expose(W*B,z)) ≈ dot(x, W*expose(B,W'z))
    @test dot(x, 2expose(W*OneBall(n),z)) ≈ dot(x, expose(2*(W*OneBall(n)),z))
end

@testset "MappedFace" begin
    n = 5; m = 3
    M = randn(m, n)
    A = OneBall(n); B = M*A
    z = randn(m)
    FA = face(A, M'*z); FB = face(B, z)
    k = rank(FA)
    c = randn(k)
    @test M*FA*c == Matrix(I,m,m)*FB*c
    @test FB*c == vec(FB)*c
end

@testset "SumAtomicSet" begin
    n = 5
    x = randn(n); z = randn(n); λ = rand(); W = randn(n,n)
    B1 = OneBall(n); B2 = λ*OneBall(n); B3 = W*B2

    @test_throws DimensionMismatch OneBall(n)+OneBall(n+1)
    @test typeof(SumAtomicSet((B1,B2))) == typeof(B1 + B2)
    @test typeof(B1+B2+B3) == typeof(SumAtomicSet((B1,B2,B3)))

    @test support((B1+B2)+B3, z) ≈ support(B1,z) + support(B2,z) + support(B3,z)
    @test support(B1+(B2+B3), z) ≈ support(B1,z) + support(B2,z) + support(B3,z)

    # With a=Σaᵢ, aᵢ∈Aᵢ, check that <a,w> = <a₁,w>+<a₂,w> for random w.
    a = expose(B1+B2+B3, z)
    a1,a2,a3 = expose(B1,z), expose(B2,z), expose(B3,z)
    @test dot(a,z) ≈ dot(a1,z) + dot(a2,z) + dot(a3,z)
    @test dot(W*a,z) ≈ dot(a,W'z)
    @test dot(W*a,z) ≈ dot(a1,W'z) + dot(a2,W'z) + dot(a3,W'z)
end

@testset "SumFace" begin
    n = 5
    A = OneBall(n); M = randn(n, n); B = M*A
    z = randn(n)
    F = face(A + B, z); Fa = face(A, z); Fb = face(B, z)
    ka = rank(Fa); kb = rank(Fb)
    ca = randn(ka); cb = randn(kb); c = vcat(ca, cb)
    @test F*c == Fa*ca + Fb*cb
end

@testset "CrossProductSet" begin
    n = 5
    x = randn(n); z = randn(n); λ = rand(); W = randn(n,n)
    B1 = OneBall(n); B2 = λ*OneBall(n); B3 = W*B2

    a1,a2,a3 = expose(B1,z), expose(B2,z), expose(B3,z)
    a = CrossProductAtom((a1,a2,a3))
    @test length(a) == length(a1)
    @test [W*a1 W*a2 W*a3] == W*a
    @test [vec(a1) vec(a2) vec(a3)] == vec(a)

    @test support((B1×B2)×B3, z) == sum([support(Bi,z) for Bi in (B1,B2,B3)])
    @test support(B1×(B2×B3), z) == sum([support(Bi,z) for Bi in (B1,B2,B3)])

    a = expose(B1×B2×B3, z)
    a1,a2,a3 = expose(B1,z), expose(B2,z), expose(B3,z)
    @test W*a == [W*a1 W*a2 W*a3]
    @test dot(a,vcat(z,z,z)) ≈ sum(a->dot(a,z), (a1,a2,a3))
    @test dot(W*a,vcat(z,z,z)) ≈ sum(a->dot(W*a,z), (a1,a2,a3))
    @test dot(a,vcat(W'z,W'z,W'z)) ≈ sum(a->dot(a,W'z), (a1,a2,a3))

    for (i, Bi) in enumerate((B1,B2,B3))
        @test (B1×B2×B3)[i] == Bi
    end
end

@testset "CrossProductFace" begin
    n = 5
    A = OneBall(n); M = randn(n, n); B = M*A
    z = randn(n)
    F = face(A×B, z); Fa = face(A, z); Fb = face(B, z)
    ka = rank(Fa); kb = rank(Fb)
    ca = randn(ka); cb = randn(kb); c = vcat(ca, cb)
    xa = Fa*ca; xb = Fb*cb; x = F*c
    @test xa == x[1]
    @test xb == x[2]

    for (i, Fi) in enumerate((Fa,Fb))
        @test typeof(F[i]) == typeof(Fi)
    end
end

@testset "Utilities" begin
    B1, B2 = OneBall(2), OneBall(3)
    @test_throws DimensionMismatch AtomicOpt.checklength([B1, B2], "")
    @test AtomicOpt.checklength([OneBall(2), OneBall(2)],"") == 2
end

@testset "NucBall" begin
    n = 5; m = 4
    x = randn(n, m); z = randn(n, m)
    A = NucBall(n, m)
    @test length(A) == n*m
    @test gauge(A,x) ≈ norm(svdvals(x), 1)
    @test support(A,z) ≈ svdvals(z)[1]
    @test dot(vec(z), expose(A,z)) ≈ support(A,z)

    F = face(A, z)
    @test convert(Matrix, I(m*n)*F) == convert(Matrix, vec(F))

end
