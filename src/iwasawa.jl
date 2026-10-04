"""
    Iwasawa <: Factorization

Matrix factorization type of the Iwasawa decomposition of a symplectic matrix `S`.
This is the return type of [`iwasawa(_)`](@ref), the corresponding matrix factorization function.

If `F::Iwasawa` is the factorization object, `N`, `A`, and `K` can be obtained
via `F.N`, `F.A`, and `F.K`, such that `S = N * A * K`. The factor `N` is a nilpotent
symplectic transformation whose leading block is unit lower triangular and whose upper
off-diagonal block vanishes, `A` is a diagonal positive-definite symplectic matrix, and `K`
is orthogonal symplectic. All three factors are unique, and each lies in a subgroup of the
symplectic group.

Iterating the decomposition produces the components `N`, `A`, and `K`, in that order.

# Examples
```jldoctest
julia> S = Symplectic(BlockForm(1), [1. 1.; 0. 1.]);

julia> issymplectic(S)
true

julia> F = iwasawa(S);

julia> isapprox(F.N * F.A * F.K, S)
true

julia> issymplectic(F.N, atol = 1e-10) && issymplectic(F.A, atol = 1e-10) && issymplectic(F.K, atol = 1e-10)
true

julia> N, A, K = F; # destructuring via iteration

julia> N == F.N && A == F.A && K == F.K
true
```
"""
struct Iwasawa{T,M<:AbstractArray{T}} <: Factorization{T}
    N::M
    A::M
    K::M
    function Iwasawa{T,M}(N, A, K) where {T,M<:AbstractArray{T}}
        require_one_based_indexing(N, A, K)
        new{T,M}(N, A, K)
    end
end
Iwasawa{T}(N::AbstractArray{T}, A::AbstractArray{T}, K::AbstractArray{T}) where {T} = Iwasawa{T,typeof(N)}(N, A, K)

# iteration for destructuring into components
Base.iterate(F::Iwasawa) = (F.N, Val(:A))
Base.iterate(F::Iwasawa, ::Val{:A}) = (F.A, Val(:K))
Base.iterate(F::Iwasawa, ::Val{:K}) = (F.K, Val(:done))
Base.iterate(F::Iwasawa, ::Val{:done}) = nothing

"""
    iwasawa(form::SymplecticForm, S::AbstractMatrix) -> Iwasawa
    iwasawa(::Type{Symplectic}, form::SymplecticForm, S::AbstractMatrix) -> Iwasawa
    iwasawa(S::Symplectic) -> Iwasawa

Compute the Iwasawa (`KAN`) decomposition of a symplectic matrix `S` and return an `Iwasawa` object.

The nilpotent factor `N`, abelian factor `A`, and orthogonal symplectic factor `K` can be
obtained via `F.N`, `F.A`, and `F.K`, such that `S = N * A * K`.

Iterating the decomposition produces the components `N`, `A`, and `K`, in that order.

# Examples
```jldoctest
julia> S = randsymplectic(Symplectic, BlockForm(3));

julia> F = iwasawa(S);

julia> isapprox(F.N * F.A * F.K, S)
true

julia> issymplectic(F.K, atol = 1e-10) && isapprox(F.K * F.K', I, atol = 1e-10)
true

julia> N, A, K = F; # destructuring via iteration

julia> N == F.N && A == F.A && K == F.K
true
```
"""
function iwasawa(form::F, x::AbstractMatrix{T}) where {F<:SymplecticForm,T<:Real}
    N, A, K = _iwasawa(form, x)
    return Iwasawa{T}(N, A, K)
end
function iwasawa(::Type{Symplectic}, form::F, x::AbstractMatrix{T}) where {F<:SymplecticForm,T<:Real}
    N, A, K = _iwasawa(form, x)
    return Iwasawa{T}(Symplectic(form, N), Symplectic(form, A), Symplectic(form, K))
end
function iwasawa(x::Symplectic{F,T,D}) where {F<:SymplecticForm,T<:Real,D<:AbstractMatrix{T}}
    N, A, K = _iwasawa(x.form, x.data)
    return Iwasawa{T}(Symplectic(x.form, N), Symplectic(x.form, A), Symplectic(x.form, K))
end

function _iwasawa(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    C::AbstractMatrix{T},
    D::AbstractMatrix{T}
) where {T<:Real}
    n = size(A, 1)
    W = A * transpose(A)
    mul!(W, B, transpose(B), one(T), one(T))
    # same Gram matrix as the pre-Iwasawa decomposition, but here it is factored
    # triangularly rather than symmetrically
    chol = cholesky(Symmetric(W), check = false)
    issuccess(chol) || throw(ArgumentError("A*Aᵀ + B*Bᵀ is not positive definite"))
    Zo = chol.L
    Zoinv = inv(Zo)
    X = Zoinv * A
    Y = Zoinv * B
    Co = C * transpose(X)
    mul!(Co, D, transpose(Y), one(T), one(T))
    κ = Vector{T}(undef, n)
    @inbounds for i in Base.OneTo(n)
        κ[i] = Zo[i,i]
    end
    # the diagonal of Zo is stripped by column scaling, so that Ao is unit lower
    # triangular and Aoinvt = Ao^{-T} stays upper triangular with unit diagonal
    Ao = Matrix{T}(undef, n, n)
    Aoinvt = Matrix{T}(undef, n, n)
    @inbounds for j in Base.OneTo(n), i in Base.OneTo(n)
        Ao[i,j] = Zo[i,j] / κ[j]
        Aoinvt[i,j] = Zoinv[j,i] * κ[j]
    end
    Co′ = Co * Zoinv
    # Co′ is symmetric analytically, so projecting onto the symmetric part before
    # multiplying by Ao keeps Aoᵀ * (Co′ * Ao) exactly symmetric, hence N exactly symplectic
    @inbounds for j in Base.OneTo(n), i in Base.OneTo(j-1)
        t = (Co′[i,j] + Co′[j,i]) / 2
        Co′[i,j] = t
        Co′[j,i] = t
    end
    mul!(Co, Co′, Ao)  # Co holds the lower-left block of N
    return Ao, Co, Aoinvt, κ, X, Y
end
function _iwasawa(form::BlockForm, x::AbstractMatrix{T}) where {T<:Real}
    n = form.n
    size(x) == (2n, 2n) || throw(ArgumentError("x must be a 2n × 2n matrix"))
    Ao, Co, Aoinvt, κ, X, Y = _iwasawa(
        @view(x[1:n, 1:n]), @view(x[1:n, n+1:2n]),
        @view(x[n+1:2n, 1:n]), @view(x[n+1:2n, n+1:2n]),
    )
    N′ = zeros(T, 2n, 2n)
    A′ = zeros(T, 2n, 2n)
    K′ = Matrix{T}(undef, 2n, 2n)
    @inbounds for j in Base.OneTo(n), i in Base.OneTo(n)
        N′[i, j] = Ao[i,j]
        N′[i+n, j] = Co[i,j]
        N′[i+n, j+n] = Aoinvt[i,j]
        K′[i, j] = X[i,j]
        K′[i, j+n] = Y[i,j]
        K′[i+n, j] = -Y[i,j]
        K′[i+n, j+n] = X[i,j]
    end
    @inbounds for i in Base.OneTo(n)
        A′[i, i] = κ[i]
        A′[i+n, i+n] = inv(κ[i])
    end
    return N′, A′, K′
end
function _iwasawa(form::PairForm, x::AbstractMatrix{T}) where {T<:Real}
    n = form.n
    size(x) == (2n, 2n) || throw(ArgumentError("x must be a 2n × 2n matrix"))
    Ao, Co, Aoinvt, κ, X, Y = _iwasawa(
        @view(x[1:2:2n-1, 1:2:2n-1]), @view(x[1:2:2n-1, 2:2:2n]),
        @view(x[2:2:2n, 1:2:2n-1]), @view(x[2:2:2n, 2:2:2n]),
    )
    N′ = zeros(T, 2n, 2n)
    A′ = zeros(T, 2n, 2n)
    K′ = Matrix{T}(undef, 2n, 2n)
    @inbounds for j in Base.OneTo(n), i in Base.OneTo(n)
        N′[2i-1, 2j-1] = Ao[i,j]
        N′[2i, 2j-1] = Co[i,j]
        N′[2i, 2j] = Aoinvt[i,j]
        K′[2i-1, 2j-1] = X[i,j]
        K′[2i-1, 2j] = Y[i,j]
        K′[2i, 2j-1] = -Y[i,j]
        K′[2i, 2j] = X[i,j]
    end
    @inbounds for i in Base.OneTo(n)
        A′[2i-1, 2i-1] = κ[i]
        A′[2i, 2i] = inv(κ[i])
    end
    return N′, A′, K′
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::Iwasawa{<:Any,<:AbstractArray})
    Base.summary(io, F); println(io)
    println(io, "N factor:")
    Base.show(io, mime, F.N)
    println(io, "\nA factor:")
    Base.show(io, mime, F.A)
    println(io, "\nK factor:")
    Base.show(io, mime, F.K)
end