@testitem "Iwasawa decomposition" begin
    using SymplecticMatrices
    using LinearAlgebra: I, Diagonal, diag, tril

    @testset "random objects" begin
        n = rand(1:5)
        J = BlockForm(n)
        Omega = PairForm(n)
        qb, pb = 1:n, n+1:2n
        qp, pp = 1:2:2n-1, 2:2:2n
        S_block = randsymplectic(J)
        S_pair = randsymplectic(Omega)

        F_block = iwasawa(J, S_block)
        N_block, A_block, K_block = iwasawa(J, S_block)
        @test F_block.N == N_block && F_block.A == A_block && F_block.K == K_block
        F_pair = iwasawa(Omega, S_pair)
        N_pair, A_pair, K_pair = iwasawa(Omega, S_pair)
        @test F_pair.N == N_pair && F_pair.A == A_pair && F_pair.K == K_pair
        @test issymplectic(J, N_block, atol = 1e-5) && issymplectic(J, A_block, atol = 1e-5) && issymplectic(J, K_block, atol = 1e-5)
        @test issymplectic(Omega, N_pair, atol = 1e-5) && issymplectic(Omega, A_pair, atol = 1e-5) && issymplectic(Omega, K_pair, atol = 1e-5)
        @test isapprox(N_block * A_block * K_block, S_block, atol = 1e-5) && isapprox(N_pair * A_pair * K_pair, S_pair, atol = 1e-5)

        # nilpotent factor should have vanishing upper block and unit lower triangular leading block
        @test isapprox(N_block[qb, pb], zeros(n, n), atol = 1e-5) && isapprox(N_pair[qp, pp], zeros(n, n), atol = 1e-5)
        @test isapprox(N_block[qb, qb], tril(N_block[qb, qb]), atol = 1e-5) && isapprox(diag(N_block[qb, qb]), ones(n), atol = 1e-5)
        @test isapprox(N_pair[qp, qp], tril(N_pair[qp, qp]), atol = 1e-5) && isapprox(diag(N_pair[qp, qp]), ones(n), atol = 1e-5)

        # nilpotent factor should be contragredient across the diagonal, with symmetric 𝒜ᵀ𝒞
        @test isapprox(transpose(N_block[qb, qb]) * N_block[pb, pb], I, atol = 1e-5) && isapprox(transpose(N_pair[qp, qp]) * N_pair[pp, pp], I, atol = 1e-5)
        @test isapprox(transpose(N_block[qb, qb]) * N_block[pb, qb], transpose(transpose(N_block[qb, qb]) * N_block[pb, qb]), atol = 1e-5)
        @test isapprox(transpose(N_pair[qp, qp]) * N_pair[pp, qp], transpose(transpose(N_pair[qp, qp]) * N_pair[pp, qp]), atol = 1e-5)

        # abelian factor should be diagonal with positive entries, lower block the inverse of the upper
        @test isapprox(A_block, Diagonal(diag(A_block)), atol = 1e-5) && isapprox(A_pair, Diagonal(diag(A_pair)), atol = 1e-5)
        @test all(i > 0 for i in diag(A_block)) && all(i > 0 for i in diag(A_pair))
        @test isapprox(A_block[qb, qb] * A_block[pb, pb], I, atol = 1e-5) && isapprox(A_pair[qp, qp] * A_pair[pp, pp], I, atol = 1e-5)

        # compact factor should be orthogonal, and of the form S(X, Y) of the U(n) embedding
        @test isapprox(inv(K_block), transpose(K_block), atol = 1e-5) && isapprox(inv(K_pair), transpose(K_pair), atol = 1e-5)
        @test isapprox(K_block[qb, qb], K_block[pb, pb], atol = 1e-5) && isapprox(K_block[qb, pb], -K_block[pb, qb], atol = 1e-5)
        @test isapprox(K_pair[qp, qp], K_pair[pp, pp], atol = 1e-5) && isapprox(K_pair[qp, pp], -K_pair[pp, qp], atol = 1e-5)

        # the triangular factor and the pre-Iwasawa factor are two square roots of the same Gram matrix
        Zo_block = N_block[qb, qb] * A_block[qb, qb]
        Zo_pair = N_pair[qp, qp] * A_pair[qp, qp]
        L, P, Q = preiwasawa(J, S_block)
        @test isapprox(Zo_block * transpose(Zo_block), P[qb, qb] * P[qb, qb], atol = 1e-5)
        @test isapprox(Zo_pair * transpose(Zo_pair), S_pair[qp, qp] * transpose(S_pair[qp, qp]) + S_pair[qp, pp] * transpose(S_pair[qp, pp]), atol = 1e-5)
    end

    @testset "symplectic type" begin
        n = rand(1:5)
        J = BlockForm(n)
        Omega = PairForm(n)
        qb, pb = 1:n, n+1:2n
        qp, pp = 1:2:2n-1, 2:2:2n
        S_block = randsymplectic(Symplectic, J)
        S_pair = randsymplectic(Symplectic, Omega)

        F_block = iwasawa(S_block)
        N_block, A_block, K_block = iwasawa(S_block)
        @test F_block.N == N_block && F_block.A == A_block && F_block.K == K_block
        F_pair = iwasawa(S_pair)
        N_pair, A_pair, K_pair = iwasawa(S_pair)
        @test F_pair.N == N_pair && F_pair.A == A_pair && F_pair.K == K_pair
        @test N_block isa Symplectic && A_block isa Symplectic && K_block isa Symplectic
        @test N_pair isa Symplectic && A_pair isa Symplectic && K_pair isa Symplectic
        G_block = iwasawa(Symplectic, J, S_block.data)
        @test G_block.N == N_block && G_block.A == A_block && G_block.K == K_block
        @test issymplectic(N_block, atol = 1e-5) && issymplectic(A_block, atol = 1e-5) && issymplectic(K_block, atol = 1e-5)
        @test issymplectic(N_pair, atol = 1e-5) && issymplectic(A_pair, atol = 1e-5) && issymplectic(K_pair, atol = 1e-5)
        @test isapprox(inv(K_block), transpose(K_block), atol = 1e-5) && isapprox(inv(K_pair), transpose(K_pair), atol = 1e-5)
        @test isapprox(A_block.data, Diagonal(diag(A_block.data)), atol = 1e-5) && all(i > 0 for i in diag(A_block.data))
        @test isapprox(A_pair.data, Diagonal(diag(A_pair.data)), atol = 1e-5) && all(i > 0 for i in diag(A_pair.data))
        @test isapprox(N_block.data[qb, qb], tril(N_block.data[qb, qb]), atol = 1e-5) && isapprox(N_pair.data[qp, qp], tril(N_pair.data[qp, qp]), atol = 1e-5)
        @test isapprox(N_block.data[qb, pb], zeros(n, n), atol = 1e-5) && isapprox(N_pair.data[qp, pp], zeros(n, n), atol = 1e-5)
        @test isapprox(N_block * A_block * K_block, S_block, atol = 1e-5) && isapprox(N_pair * A_pair * K_pair, S_pair, atol = 1e-5)
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
        N, A, K = iwasawa(J, K_block)
        @test isapprox(N, I, atol = 1e-5) && isapprox(A, I, atol = 1e-5) && isapprox(K, K_block, atol = 1e-5)
        N, A, K = iwasawa(Omega, K_pair)
        @test isapprox(N, I, atol = 1e-5) && isapprox(A, I, atol = 1e-5) && isapprox(K, K_pair, atol = 1e-5)

        # S in A: the abelian factor absorbs everything
        kappa = rand(n) .+ 0.5
        S_block = zeros(2n, 2n); S_pair = zeros(2n, 2n)
        for i in Base.OneTo(n)
            S_block[i, i] = kappa[i]; S_block[i+n, i+n] = inv(kappa[i])
            S_pair[2i-1, 2i-1] = kappa[i]; S_pair[2i, 2i] = inv(kappa[i])
        end
        @test issymplectic(J, S_block, atol = 1e-5) && issymplectic(Omega, S_pair, atol = 1e-5)
        N, A, K = iwasawa(J, S_block)
        @test isapprox(N, I, atol = 1e-5) && isapprox(A, S_block, atol = 1e-5) && isapprox(K, I, atol = 1e-5)
        N, A, K = iwasawa(Omega, S_pair)
        @test isapprox(N, I, atol = 1e-5) && isapprox(A, S_pair, atol = 1e-5) && isapprox(K, I, atol = 1e-5)

        # S in the lens subgroup T^(l) ⊂ N: the nilpotent factor absorbs everything
        M = rand(n, n)
        C0 = M + transpose(M)
        S_block = Matrix{Float64}(I, 2n, 2n); S_block[pb, qb] = C0
        S_pair = Matrix{Float64}(I, 2n, 2n); S_pair[pp, qp] = C0
        @test issymplectic(J, S_block, atol = 1e-5) && issymplectic(Omega, S_pair, atol = 1e-5)
        N, A, K = iwasawa(J, S_block)
        @test isapprox(N, S_block, atol = 1e-5) && isapprox(A, I, atol = 1e-5) && isapprox(K, I, atol = 1e-5)
        N, A, K = iwasawa(Omega, S_pair)
        @test isapprox(N, S_pair, atol = 1e-5) && isapprox(A, I, atol = 1e-5) && isapprox(K, I, atol = 1e-5)

        # S in N with nontrivial leading block: still absorbed entirely by the nilpotent factor
        Aa = Matrix{Float64}(I, n, n) + tril(M, -1)
        Cc = transpose(inv(Aa)) * C0
        S_block = zeros(2n, 2n); S_pair = zeros(2n, 2n)
        S_block[qb, qb] = Aa; S_block[pb, qb] = Cc; S_block[pb, pb] = transpose(inv(Aa))
        S_pair[qp, qp] = Aa; S_pair[pp, qp] = Cc; S_pair[pp, pp] = transpose(inv(Aa))
        @test issymplectic(J, S_block, atol = 1e-5) && issymplectic(Omega, S_pair, atol = 1e-5)
        N, A, K = iwasawa(J, S_block)
        @test isapprox(N, S_block, atol = 1e-5) && isapprox(A, I, atol = 1e-5) && isapprox(K, I, atol = 1e-5)
        N, A, K = iwasawa(Omega, S_pair)
        @test isapprox(N, S_pair, atol = 1e-5) && isapprox(A, I, atol = 1e-5) && isapprox(K, I, atol = 1e-5)

        # S in GL(n,ℝ) ∩ Π(n): the compact factor drops out, but N and A share the rest
        Ao = M * transpose(M) + n * Matrix{Float64}(I, n, n)
        S_block = zeros(2n, 2n); S_block[qb, qb] = Ao; S_block[pb, pb] = inv(Ao)
        S_pair = zeros(2n, 2n); S_pair[qp, qp] = Ao; S_pair[pp, pp] = inv(Ao)
        @test issymplectic(J, S_block, atol = 1e-5) && issymplectic(Omega, S_pair, atol = 1e-5)
        N, A, K = iwasawa(J, S_block)
        @test isapprox(K, I, atol = 1e-5) && isapprox(N[pb, qb], zeros(n, n), atol = 1e-5)
        @test isapprox((N[qb, qb] * A[qb, qb]) * transpose(N[qb, qb] * A[qb, qb]), Ao * Ao, atol = 1e-5)
        N, A, K = iwasawa(Omega, S_pair)
        @test isapprox(K, I, atol = 1e-5) && isapprox(N[pp, qp], zeros(n, n), atol = 1e-5)
        @test isapprox((N[qp, qp] * A[qp, qp]) * transpose(N[qp, qp] * A[qp, qp]), Ao * Ao, atol = 1e-5)

        # n = 1: closed form of eq. (4.22) of quant-ph/9509002
        S = randsymplectic(BlockForm(1))
        a, b, c, d = S[1,1], S[1,2], S[2,1], S[2,2]
        xi = (a * c + b * d) / (a^2 + b^2)
        eta = log(a^2 + b^2)
        phi = 2 * angle(a - im * b)
        N, A, K = iwasawa(BlockForm(1), S)
        @test isapprox(N, [1 0; xi 1], atol = 1e-5)
        @test isapprox(A, [exp(eta/2) 0; 0 exp(-eta/2)], atol = 1e-5)
        @test isapprox(K, [cos(phi/2) -sin(phi/2); sin(phi/2) cos(phi/2)], atol = 1e-5)
        N_pair, A_pair, K_pair = iwasawa(PairForm(1), S)
        @test isapprox(N_pair, N, atol = 1e-5) && isapprox(A_pair, A, atol = 1e-5) && isapprox(K_pair, K, atol = 1e-5)

        # for n = 1 the O(1) ambiguity is fixed identically by both gauges
        L, P, Q = preiwasawa(BlockForm(1), S)
        @test isapprox(N, L, atol = 1e-5) && isapprox(A, P, atol = 1e-5) && isapprox(K, Q, atol = 1e-5)
    end
end