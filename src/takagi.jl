"""
    Takagi <: Factorization

Matrix factorization type of the Takagi decomposition of a complex symmetric matrix `x`.
This is the return type of [`takagi(_)`](@ref), the corresponding matrix factorization function.

If `F::Takagi` is the factorization object, then `Q` and `S` can be obtained
via `F.Q` and `F.S`, such that `x = Q * Diagonal(S) * transpose(Q)`. The factor `Q` is unitary,
and `S` contains the singular values of `x` in descending order.

Iterating the decomposition produces the components `Q` and `S`, in that order.

# Examples
```jldoctest
julia> x = [1.0+2.0im 0.5im; 0.5im 3.0];

julia> F = takagi(x);

julia> isapprox(F.Q * Diagonal(F.S) * transpose(F.Q), x)
true

julia> isapprox(F.Q' * F.Q, I)
true

julia> Q, S = F; # destructuring via iteration

julia> Q == F.Q && S == F.S
true
```
"""
struct Takagi{T,M<:AbstractArray{T},N<:AbstractVector{<:Real}} <: Factorization{T}
    Q::M
    S::N
    function Takagi{T,M,N}(Q, S) where {T,M<:AbstractArray{T},N<:AbstractVector{<:Real}}
        require_one_based_indexing(Q, S)
        new{T,M,N}(Q, S)
    end
end
Takagi{T}(Q::AbstractArray{T}, S::AbstractVector{<:Real}) where {T} = Takagi{T,typeof(Q),typeof(S)}(Q,S)

# iteration for destructuring into components
Base.iterate(F::Takagi) = (F.Q, Val(:S))
Base.iterate(F::Takagi, ::Val{:S}) = (F.S, Val(:done))
Base.iterate(F::Takagi, ::Val{:done}) = nothing

"""
    takagi(x::AbstractMatrix) -> Takagi

Compute the Takagi decomposition of a complex symmetric matrix `x` and return a `Takagi` object.

The unitary factor `Q` and the singular values `S` can be obtained via `F.Q` and `F.S`,
such that `x = Q * Diagonal(S) * transpose(Q)`. Only the upper triangle of `x` is referenced.

Iterating the decomposition produces the components `Q` and `S`, in that order.

# Examples
```jldoctest
julia> x = rand(ComplexF64, 3, 3); x = x + transpose(x);

julia> F = takagi(x);

julia> isapprox(F.Q * Diagonal(F.S) * transpose(F.Q), x)
true

julia> issorted(F.S, rev = true)
true

julia> Q, S = F; # destructuring via iteration

julia> Q == F.Q && S == F.S
true
```
"""
function takagi(x::AbstractMatrix{T}) where {T<:Union{Real,Complex}}
    Q, S = _takagi(x)
    return Takagi{eltype(Q)}(Q, S)
end
function _takagi(x::AbstractMatrix{T}) where {T<:Real}
    # for real x, the unitary Uᴴ * conj(V) of the complex method is the diagonal matrix of
    # eigenvalue signs, so its square root is known in closed form and no SVD or Schur form is needed
    vals, vecs = eigen(Symmetric(x), sortby = λ -> -abs(λ))
    Q = similar(vecs, complex(eltype(vecs)))
    @inbounds for j in axes(vecs, 2)
        neg = vals[j] < zero(eltype(vals))
        for i in axes(vecs, 1)
            v = vecs[i,j]
            Q[i,j] = neg ? complex(zero(v), v) : complex(v)
        end
        vals[j] = abs(vals[j])
    end
    return Q, vals
end
function _takagi(x::AbstractMatrix{T}) where {T<:Complex}
    fact = svd!(Matrix{float(T)}(Symmetric(x)))
    U = fact.U
    # W = Uᴴ * conj(V) is unitary, block diagonal over equal singular values, and symmetric on
    # every block of nonzero singular value, so that Q = U * sqrt(W) gives x = Q * Diagonal(S) * Qᵀ
    W = transpose(U) * fact.V
    conj!(W)
    # W is normal, so its Schur form is diagonal
    sF = schur!(W)
    vals = sF.values
    n = length(vals)
    # midpoint of the largest arc of the unit circle that is free of eigenvalues of W
    θ = sort!(mod2pi.(angle.(vals)))
    period = 2 * oftype(first(θ), π)
    gap, mid = zero(period), zero(period)
    @inbounds for j in Base.OneTo(n)
        g = (j < n ? θ[j+1] : θ[1] + period) - θ[j]
        g > gap && ((gap, mid) = (g, θ[j] + g / 2))
    end
    # rotating the spectrum by cis(shift) moves this midpoint onto the branch cut of sqrt, so that
    # numerically degenerate eigenvalues receive the same root and sqrt(W) remains a function of W
    shift = mod2pi(-mid) - period / 2
    rot, unrot = cis(shift), cis(-shift / 2)
    mul!(W, U, sF.Z)
    @inbounds for j in Base.OneTo(n)
        r = sqrt(vals[j] * rot) * unrot
        for i in Base.OneTo(n)
            W[i,j] *= r
        end
    end
    Q = mul!(U, W, adjoint(sF.Z))
    return Q, fact.S
end
function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::Takagi{<:Any,<:AbstractArray,<:AbstractVector})
    Base.summary(io, F); println(io)
    println(io, "Q factor:")
    Base.show(io, mime, F.Q)
    println(io, "\nsingular values:")
    Base.show(io, mime, F.S)
end