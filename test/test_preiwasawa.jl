@testitem "pre-Iwasawa decomposition" begin
    using SymplecticMatrices
    using LinearAlgebra: I, Symmetric, eigvals

    @testset "random objects" begin
        n = rand(1:5)
        J = BlockForm(n)
        Omega = PairForm(n)
        qb, pb = 1:n, n+1:2n
        qp, pp = 1:2:2n-1, 2:2:2n
        S_block = randsymplectic(J)
        S_pair = randsymplectic(Omega)

        F_block = preiwasawa(J, S_block)
        L_block, P_block, Q_block = preiwasawa(J, S_block)
        @test F_block.L == L_block && F_block.P == P_block && F_block.Q == Q_block
        F_pair = preiwasawa(Omega, S_pair)
        L_pair, P_pair, Q_pair = preiwasawa(Omega, S_pair)
        @test F_pair.L == L_pair && F_pair.P == P_pair && F_pair.Q == Q_pair
        @test issymplectic(J, L_block, atol = 1e-5) && issymplectic(J, P_block, atol = 1e-5) && issymplectic(J, Q_block, atol = 1e-5)
        @test issymplectic(Omega, L_pair, atol = 1e-5) && issymplectic(Omega, P_pair, atol = 1e-5) && issymplectic(Omega, Q_pair, atol = 1e-5)
        @test isapprox(L_block * P_block * Q_block, S_block, atol = 1e-5) && isapprox(L_pair * P_pair * Q_pair, S_pair, atol = 1e-5)

        # lens factor should have unit diagonal blocks, vanishing upper block, symmetric lower block
        @test isapprox(L_block[qb, qb], I, atol = 1e-5) && isapprox(L_block[pb, pb], I, atol = 1e-5)
        @test isapprox(L_pair[qp, qp], I, atol = 1e-5) && isapprox(L_pair[pp, pp], I, atol = 1e-5)
        @test isapprox(L_block[qb, pb], zeros(n, n), atol = 1e-5) && isapprox(L_pair[qp, pp], zeros(n, n), atol = 1e-5)
        @test isapprox(L_block[pb, qb], transpose(L_block[pb, qb]), atol = 1e-5) && isapprox(L_pair[pp, qp], transpose(L_pair[pp, qp]), atol = 1e-5)

        # middle factor should be block diagonal, symmetric positive definite, lower block the inverse of the upper
        @test isapprox(P_block[qb, pb], zeros(n, n), atol = 1e-5) && isapprox(P_block[pb, qb], zeros(n, n), atol = 1e-5)
        @test isapprox(P_pair[qp, pp], zeros(n, n), atol = 1e-5) && isapprox(P_pair[pp, qp], zeros(n, n), atol = 1e-5)
        @test isapprox(P_block[qb, qb] * P_block[pb, pb], I, atol = 1e-5) && isapprox(P_pair[qp, qp] * P_pair[pp, pp], I, atol = 1e-5)
        @test isapprox(P_block, transpose(P_block), atol = 1e-5) && all(i > 0 for i in eigvals(Symmetric(P_block[qb, qb])))
        @test isapprox(P_pair, transpose(P_pair), atol = 1e-5) && all(i > 0 for i in eigvals(Symmetric(P_pair[qp, qp])))

        # compact factor should be orthogonal, and of the form S(X, Y) of the U(n) embedding
        @test isapprox(inv(Q_block), transpose(Q_block), atol = 1e-5) && isapprox(inv(Q_pair), transpose(Q_pair), atol = 1e-5)
        @test isapprox(Q_block[qb, qb], Q_block[pb, pb], atol = 1e-5) && isapprox(Q_block[qb, pb], -Q_block[pb, qb], atol = 1e-5)
        @test isapprox(Q_pair[qp, qp], Q_pair[pp, pp], atol = 1e-5) && isapprox(Q_pair[qp, pp], -Q_pair[pp, qp], atol = 1e-5)
    end

    @testset "symplectic type" begin
        n = rand(1:5)
        J = BlockForm(n)
        Omega = PairForm(n)
        qb, pb = 1:n, n+1:2n
        qp, pp = 1:2:2n-1, 2:2:2n
        S_block = randsymplectic(Symplectic, J)
        S_pair = randsymplectic(Symplectic, Omega)

        F_block = preiwasawa(S_block)
        L_block, P_block, Q_block = preiwasawa(S_block)
        @test F_block.L == L_block && F_block.P == P_block && F_block.Q == Q_block
        F_pair = preiwasawa(S_pair)
        L_pair, P_pair, Q_pair = preiwasawa(S_pair)
        @test F_pair.L == L_pair && F_pair.P == P_pair && F_pair.Q == Q_pair
        @test L_block isa Symplectic && P_block isa Symplectic && Q_block isa Symplectic
        @test L_pair isa Symplectic && P_pair isa Symplectic && Q_pair isa Symplectic
        G_block = preiwasawa(Symplectic, J, S_block.data)
        @test G_block.L == L_block && G_block.P == P_block && G_block.Q == Q_block
        @test issymplectic(L_block, atol = 1e-5) && issymplectic(P_block, atol = 1e-5) && issymplectic(Q_block, atol = 1e-5)
        @test issymplectic(L_pair, atol = 1e-5) && issymplectic(P_pair, atol = 1e-5) && issymplectic(Q_pair, atol = 1e-5)
        @test isapprox(inv(Q_block), transpose(Q_block), atol = 1e-5) && isapprox(inv(Q_pair), transpose(Q_pair), atol = 1e-5)
        @test isapprox(P_block, transpose(P_block), atol = 1e-5) && all(i > 0 for i in eigvals(Symmetric(P_block.data[qb, qb])))
        @test isapprox(P_pair, transpose(P_pair), atol = 1e-5) && all(i > 0 for i in eigvals(Symmetric(P_pair.data[qp, qp])))
        @test isapprox(L_block.data[pb, qb], transpose(L_block.data[pb, qb]), atol = 1e-5) && isapprox(L_pair.data[pp, qp], transpose(L_pair.data[pp, qp]), atol = 1e-5)
        @test isapprox(L_block * P_block * Q_block, S_block, atol = 1e-5) && isapprox(L_pair * P_pair * Q_pair, S_pair, atol = 1e-5)
    end

    @testset "limiting cases" begin
        n = rand(2:5)
        J = BlockForm(n)
        Omega = PairForm(n)
        qb, pb = 1:n, n+1:2n
        qp, pp = 1:2:2n-1, 2:2:2n

        # S in K(n): the compact factor absorbs everything
        K_block = Matrix(givens(J, rand(1:n-1), 2pi * rand()))
        K_pair = Matrix(givens(Omega, rand(1:n-1), 2pi * rand()))
        L, P, Q = preiwasawa(J, K_block)
        @test isapprox(L, I, atol = 1e-5) && isapprox(P, I, atol = 1e-5) && isapprox(Q, K_block, atol = 1e-5)
        L, P, Q = preiwasawa(Omega, K_pair)
        @test isapprox(L, I, atol = 1e-5) && isapprox(P, I, atol = 1e-5) && isapprox(Q, K_pair, atol = 1e-5)

        # S in GL(n,ℝ) ∩ Π(n): the middle factor absorbs everything
        M = rand(n, n)
        Ao = M * transpose(M) + n * Matrix{Float64}(I, n, n)
        S_block = zeros(2n, 2n); S_block[qb, qb] = Ao; S_block[pb, pb] = inv(Ao)
        S_pair = zeros(2n, 2n); S_pair[qp, qp] = Ao; S_pair[pp, pp] = inv(Ao)
        @test issymplectic(J, S_block, atol = 1e-5) && issymplectic(Omega, S_pair, atol = 1e-5)
        L, P, Q = preiwasawa(J, S_block)
        @test isapprox(L, I, atol = 1e-5) && isapprox(P, S_block, atol = 1e-5) && isapprox(Q, I, atol = 1e-5)
        L, P, Q = preiwasawa(Omega, S_pair)
        @test isapprox(L, I, atol = 1e-5) && isapprox(P, S_pair, atol = 1e-5) && isapprox(Q, I, atol = 1e-5)

        # S in the lens subgroup: the lens factor absorbs everything
        C0 = M + transpose(M)
        S_block = Matrix{Float64}(I, 2n, 2n); S_block[pb, qb] = C0
        S_pair = Matrix{Float64}(I, 2n, 2n); S_pair[pp, qp] = C0
        @test issymplectic(J, S_block, atol = 1e-5) && issymplectic(Omega, S_pair, atol = 1e-5)
        L, P, Q = preiwasawa(J, S_block)
        @test isapprox(L, S_block, atol = 1e-5) && isapprox(P, I, atol = 1e-5) && isapprox(Q, I, atol = 1e-5)
        L, P, Q = preiwasawa(Omega, S_pair)
        @test isapprox(L, S_pair, atol = 1e-5) && isapprox(P, I, atol = 1e-5) && isapprox(Q, I, atol = 1e-5)

        # n = 1: closed form of eq. (4.22) of quant-ph/9509002
        S = randsymplectic(BlockForm(1))
        a, b, c, d = S[1,1], S[1,2], S[2,1], S[2,2]
        xi = (a * c + b * d) / (a^2 + b^2)
        eta = log(a^2 + b^2)
        phi = 2 * angle(a - im * b)
        L, P, Q = preiwasawa(BlockForm(1), S)
        @test isapprox(L, [1 0; xi 1], atol = 1e-5)
        @test isapprox(P, [exp(eta/2) 0; 0 exp(-eta/2)], atol = 1e-5)
        @test isapprox(Q, [cos(phi/2) -sin(phi/2); sin(phi/2) cos(phi/2)], atol = 1e-5)
        L_pair, P_pair, Q_pair = preiwasawa(PairForm(1), S)
        @test isapprox(L_pair, L, atol = 1e-5) && isapprox(P_pair, P, atol = 1e-5) && isapprox(Q_pair, Q, atol = 1e-5)
    end
end