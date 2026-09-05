"""
    PreIwasawa <: Factorization

Matrix factorization type of the pre-Iwasawa decomposition of a symplectic matrix `S`.
This is the return type of [`preiwasawa(_)`](@ref), the corresponding matrix factorization function.

If `F::PreIwasawa` is the factorization object, `L`, `P`, and `Q` can be obtained
via `F.L`, `F.P`, and `F.Q`, such that `S = L * P * Q`. The factor `L` is a symplectic lens
transformation with unit diagonal blocks and symmetric off-diagonal block, `P` is a
block-diagonal symmetric positive-definite symplectic matrix, and `Q` is orthogonal
symplectic. All three factors are unique.

Iterating the decomposition produces the components `L`, `P`, and `Q`, in that order.

# Examples
```jldoctest
julia> S = Symplectic(BlockForm(1), [1. 1.; 0. 1.]);

julia> issymplectic(S)
true

julia> F = preiwasawa(S);

julia> isapprox(F.L * F.P * F.Q, S)
true

julia> issymplectic(F.L, atol = 1e-10) && issymplectic(F.P, atol = 1e-10) && issymplectic(F.Q, atol = 1e-10)
true

julia> L, P, Q = F; # destructuring via iteration

julia> L == F.L && P == F.P && Q == F.Q
true
```
"""
struct PreIwasawa{T,M<:AbstractArray{T}} <: Factorization{T}
    L::M
    P::M
    Q::M
    function PreIwasawa{T,M}(L, P, Q) where {T,M<:AbstractArray{T}}
        require_one_based_indexing(L, P, Q)
        new{T,M}(L, P, Q)
    end
end
PreIwasawa{T}(L::AbstractArray{T}, P::AbstractArray{T}, Q::AbstractArray{T}) where {T} = PreIwasawa{T,typeof(L)}(L, P, Q)

# iteration for destructuring into components
Base.iterate(F::PreIwasawa) = (F.L, Val(:P))
Base.iterate(F::PreIwasawa, ::Val{:P}) = (F.P, Val(:Q))
Base.iterate(F::PreIwasawa, ::Val{:Q}) = (F.Q, Val(:done))
Base.iterate(F::PreIwasawa, ::Val{:done}) = nothing

"""
    preiwasawa(form::SymplecticForm, S::AbstractMatrix) -> PreIwasawa
    preiwasawa(::Type{Symplectic}, form::SymplecticForm, S::AbstractMatrix) -> PreIwasawa
    preiwasawa(S::Symplectic) -> PreIwasawa

Compute the pre-Iwasawa decomposition of a symplectic matrix `S` and return a `PreIwasawa` object.

The lens factor `L`, positive-definite factor `P`, and orthogonal symplectic factor `Q` can be
obtained via `F.L`, `F.P`, and `F.Q`, such that `S = L * P * Q`.

Iterating the decomposition produces the components `L`, `P`, and `Q`, in that order.

# Examples
```jldoctest
julia> S = randsymplectic(Symplectic, BlockForm(3));

julia> F = preiwasawa(S);

julia> isapprox(F.L * F.P * F.Q, S)
true

julia> issymplectic(F.Q, atol = 1e-10) && isapprox(F.Q * F.Q', I, atol = 1e-10)
true

julia> L, P, Q = F; # destructuring via iteration

julia> L == F.L && P == F.P && Q == F.Q
true
```
"""
function preiwasawa(form::F, x::AbstractMatrix{T}) where {F<:SymplecticForm,T<:Real}
    L, P, Q = _preiwasawa(form, x)
    return PreIwasawa{T}(L, P, Q)
end
function preiwasawa(::Type{Symplectic}, form::F, x::AbstractMatrix{T}) where {F<:SymplecticForm,T<:Real}
    L, P, Q = _preiwasawa(form, x)
    return PreIwasawa{T}(Symplectic(form, L), Symplectic(form, P), Symplectic(form, Q))
end
function preiwasawa(x::Symplectic{F,T,D}) where {F<:SymplecticForm,T<:Real,D<:AbstractMatrix{T}}
    L, P, Q = _preiwasawa(x.form, x.data)
    return PreIwasawa{T}(Symplectic(x.form, L), Symplectic(x.form, P), Symplectic(x.form, Q))
end

function _preiwasawa(
    A::AbstractMatrix{T}, 
    B::AbstractMatrix{T}, 
    C::AbstractMatrix{T}, 
    D::AbstractMatrix{T}
) where {T<:Real}
    n = size(A, 1)
    W = A * transpose(A)
    mul!(W, B, transpose(B), one(T), one(T))
    vals, vecs = eigen(Symmetric(W))
    first(vals) > zero(T) || throw(ArgumentError("A*Aᵀ + B*Bᵀ is not positive definite"))
    # both Ao and its inverse come from the single eigendecomposition of Ao^2
    @inbounds for j in Base.OneTo(n)
        r = sqrt(vals[j])
        for i in Base.OneTo(n)
            W[i,j] = vecs[i,j] * r
        end
    end
    Ao = W * transpose(vecs)
    @inbounds for j in Base.OneTo(n)
        r = inv(sqrt(vals[j]))
        for i in Base.OneTo(n)
            W[i,j] = vecs[i,j] * r
        end
    end
    Aoinv = W * transpose(vecs)
    X = Aoinv * A
    Y = Aoinv * B
    Co = C * transpose(X)
    mul!(Co, D, transpose(Y), one(T), one(T))
    Co′ = Co * Aoinv
    # Co′ is symmetric analytically, so projecting onto the symmetric part keeps
    # the lens factor exactly symplectic
    @inbounds for j in Base.OneTo(n), i in Base.OneTo(j-1)
        t = (Co′[i,j] + Co′[j,i]) / 2
        Co′[i,j] = t
        Co′[j,i] = t
    end
    return Co′, Ao, Aoinv, X, Y
end
function _preiwasawa(form::BlockForm, x::AbstractMatrix{T}) where {T<:Real}
    n = form.n
    size(x) == (2n, 2n) || throw(ArgumentError("x must be a 2n × 2n matrix"))
    Co′, Ao, Aoinv, X, Y = _preiwasawa(
        @view(x[1:n, 1:n]), @view(x[1:n, n+1:2n]),
        @view(x[n+1:2n, 1:n]), @view(x[n+1:2n, n+1:2n]),
    )
    L′ = Matrix{T}(I, 2n, 2n)
    P′ = zeros(T, 2n, 2n)
    Q′ = Matrix{T}(undef, 2n, 2n)
    @inbounds for j in Base.OneTo(n), i in Base.OneTo(n)
        L′[i+n, j] = Co′[i,j]
        P′[i, j] = Ao[i,j]
        P′[i+n, j+n] = Aoinv[i,j]
        Q′[i, j] = X[i,j]
        Q′[i, j+n] = Y[i,j]
        Q′[i+n, j] = -Y[i,j]
        Q′[i+n, j+n] = X[i,j]
    end
    return L′, P′, Q′
end
function _preiwasawa(form::PairForm, x::AbstractMatrix{T}) where {T<:Real}
    n = form.n
    size(x) == (2n, 2n) || throw(ArgumentError("x must be a 2n × 2n matrix"))
    Co′, Ao, Aoinv, X, Y = _preiwasawa(
        x[1:2:2n-1, 1:2:2n-1], x[1:2:2n-1, 2:2:2n],
        x[2:2:2n, 1:2:2n-1], x[2:2:2n, 2:2:2n],
    )
    L′ = Matrix{T}(I, 2n, 2n)
    P′ = zeros(T, 2n, 2n)
    Q′ = Matrix{T}(undef, 2n, 2n)
    @inbounds for j in Base.OneTo(n), i in Base.OneTo(n)
        L′[2i, 2j-1] = Co′[i,j]
        P′[2i-1, 2j-1] = Ao[i,j]
        P′[2i, 2j] = Aoinv[i,j]
        Q′[2i-1, 2j-1] = X[i,j]
        Q′[2i-1, 2j] = Y[i,j]
        Q′[2i, 2j-1] = -Y[i,j]
        Q′[2i, 2j] = X[i,j]
    end
    return L′, P′, Q′
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::PreIwasawa{<:Any,<:AbstractArray})
    Base.summary(io, F); println(io)
    println(io, "L factor:")
    Base.show(io, mime, F.L)
    println(io, "\nP factor:")
    Base.show(io, mime, F.P)
    println(io, "\nQ factor:")
    Base.show(io, mime, F.Q)
end

