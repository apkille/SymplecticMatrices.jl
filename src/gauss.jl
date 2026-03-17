"""
    SymplecticGauss <: AbstractMatrix

Matrix representation type of a symplectic Gauss transformation on a symplectic basis.
This is the return type of [`gauss(_)`](@ref), the corresponding transformation matrix function.

If `G::SymplecticGauss` is the transformation object, the mode `k` and parameters `c` and `d` can be obtained
via `G.k`, `G.c`, and `G.d`.

# Examples
```jldoctest
julia> G = gauss(PairForm(3), 2, 2.0, 3.0)
6×6 SymplecticGauss{PairForm{Int64}, Int64, Float64}:
 2.0   ⋅    ⋅   3.0   ⋅    ⋅ 
  ⋅   0.5   ⋅    ⋅    ⋅    ⋅ 
  ⋅   3.0  2.0   ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅   0.5   ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅   1.0   ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅   1.0

julia> isapprox(G * inv(G), I)
true
```
"""
struct SymplecticGauss{F<:SymplecticForm,N<:Int,T} <: AbstractMatrix{T}
    form::F
    k::N
    c::T
    d::T
    SymplecticGauss(form::F, k::N, c::T, d::T) where {F<:SymplecticForm,N<:Int,T} = new{F,N,T}(form, k, c, d)
end
SymplecticGauss(x::SymplecticGauss) = x

function gauss(form::SymplecticForm, k::Int, c::T, d::T) where {T}
    2 <= k <= form.n || throw(ArgumentError("k must be between 2 and n."))
    return SymplecticGauss(form, k, c, d)
end

function Base.Matrix(x::SymplecticGauss{F,N,T}) where {F<:BlockForm,N<:Int,T}
    n, k, c, d = (x.form).n, x.k, x.c, x.d
    M = Matrix{T}(I, 2n, 2n)
    M[k-1, k-1] = c
    M[k, k] = c
    M[k-1, n+k] = d
    M[k, n+k-1] = d
    M[n+k-1, n+k-1] = inv(c)
    M[n+k, n+k] = inv(c)
    return M
end
function Base.Matrix(x::SymplecticGauss{F,N,T}) where {F<:PairForm,N<:Int,T}
    n, k, c, d = (x.form).n, x.k, x.c, x.d
    M = Matrix{T}(I, 2n, 2n)
    M[2k-3, 2k-3] = c
    M[2k-1, 2k-1] = c
    M[2k-3, 2k] = d
    M[2k-1, 2k-2] = d
    M[2k-2, 2k-2] = inv(c)
    M[2k, 2k] = inv(c)
    return M
end
Base.Array(x::SymplecticGauss) = Matrix(x)
function Base.AbstractMatrix{T1}(x::SymplecticGauss{F,N,T2}) where {F<:BlockForm,N<:Int,T1,T2}
    n, k, c, d = (x.form).n, x.k, x.c, x.d
    M = Matrix{T1}(I, 2n, 2n)
    M[k-1, k-1] = c
    M[k, k] = c
    M[k-1, n+k] = d
    M[k, n+k-1] = d
    M[n+k-1, n+k-1] = inv(c)
    M[n+k, n+k] = inv(c)
    return M
end
function Base.AbstractMatrix{T1}(x::SymplecticGauss{F,N,T2}) where {F<:PairForm,N<:Int,T1,T2}
    n, k, c, d = (x.form).n, x.k, x.c, x.d
    M = Matrix{T1}(I, 2n, 2n)
    M[2k-3, 2k-3] = c
    M[2k-1, 2k-1] = c
    M[2k-3, 2k] = d
    M[2k-1, 2k-2] = d
    M[2k-2, 2k-2] = inv(c)
    M[2k, 2k] = inv(c)
    return M
end

Base.checkbounds(x::SymplecticGauss, i::Int, j::Int) = (form = x.form; i <= 2 * form.n && j <= 2 * form.n)
Base.checkbounds(x::SymplecticGauss, i::Int) = (form = x.form; i <= (2 * form.n)^2)
Base.isequal(x::SymplecticGauss, y::SymplecticGauss) = x.form == y.form && x.k == y.k && x.c == y.c && x.d == y.d
Base.isapprox(x::SymplecticGauss, y::SymplecticGauss) = x.form == y.form && isapprox(x.k, y.k) && isapprox(x.c, y.c) && isapprox(x.d, y.d)
Base.size(x::SymplecticGauss) = (form = x.form; (2 * form.n, 2 * form.n))
Base.size(x::SymplecticGauss, dim::Int) = (form = x.form; 2 * form.n)
Base.length(x::SymplecticGauss) = prod(size(x))
Base.axes(x::SymplecticGauss) = (form = x.form; (Base.OneTo(2 * form.n), Base.OneTo(2 * form.n)))
Base.eltype(x::SymplecticGauss) = typeof(x.c)

Base.@propagate_inbounds function Base.getindex(x::SymplecticGauss{F,N,T}, i::Int, j::Int) where {F<:BlockForm,N<:Int,T}
    @boundscheck checkbounds(x, i, j)
    n, k, c, d = (x.form).n, x.k, x.c, x.d
    if i == j
        if i == k - 1 || i == k
            return c
        elseif i == n + k - 1 || i == n + k
            return inv(c)
        else
            return oneunit(T)
        end
    elseif i == k - 1 && j == n + k
        return d
    elseif i == k && j == n + k - 1
        return d
    else
        return zero(T)
    end
end
Base.@propagate_inbounds function Base.getindex(x::SymplecticGauss{F,N,T}, i::Int, j::Int) where {F<:PairForm,N<:Int,T}
    @boundscheck checkbounds(x, i, j)
    n, k, c, d = (x.form).n, x.k, x.c, x.d
    if i == j
        if i == 2k - 3 || i == 2k - 1
            return c
        elseif i == 2k - 2 || i == 2k
            return inv(c)
        else
            return oneunit(T)
        end
    elseif i == 2k - 3 && j == 2k
        return d
    elseif i == 2k - 1 && j == 2k - 2
        return d
    else
        return zero(T)
    end
end
Base.@propagate_inbounds function Base.getindex(x::SymplecticGauss, ind::Int)
    @boundscheck checkbounds(x, ind)
    n2 = 2 * x.form.n
    c_idx, r = divrem(ind - 1, n2)
    return x[r + 1, c_idx + 1]
end

LinearAlgebra.inv(x::SymplecticGauss) = SymplecticGauss(x.form, x.k, inv(x.c), -x.d)
Base.copy(x::SymplecticGauss) = SymplecticGauss(x.form, copy(x.k), copy(x.c), copy(x.d))

@inline function Base.copyto!(dest::AbstractMatrix, src::SymplecticGauss{F,N,T}) where {F<:BlockForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(dest)
    size(dest, 1) == size(dest, 2) || throw(ArgumentError("cannot copy a SymplecticGauss object to a non-square matrix."))
    n, k, c, d = (src.form).n, src.k, src.c, src.d
    fill!(dest, zero(T))
    @inbounds for i in axes(dest, 1)
        dest[i, i] = oneunit(T)
    end
    dest[k-1, k-1] = c
    dest[k, k] = c
    dest[k-1, n+k] = d
    dest[k, n+k-1] = d
    dest[n+k-1, n+k-1] = inv(c)
    dest[n+k, n+k] = inv(c)
    return dest
end
@inline function Base.copyto!(dest::Symplectic, src::SymplecticGauss{F,N,T}) where {F<:BlockForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(dest)
    @assert dest.form == src.form
    size(dest, 1) == size(dest, 2) || throw(ArgumentError("cannot copy a SymplecticGauss object to a non-square matrix."))
    n, k, c, d = (src.form).n, src.k, src.c, src.d
    fill!(dest, zero(T))
    @inbounds for i in axes(dest, 1)
        dest[i, i] = oneunit(T)
    end
    dest[k-1, k-1] = c
    dest[k, k] = c
    dest[k-1, n+k] = d
    dest[k, n+k-1] = d
    dest[n+k-1, n+k-1] = inv(c)
    dest[n+k, n+k] = inv(c)
    return dest
end
@inline function Base.copyto!(dest::AbstractMatrix, src::SymplecticGauss{F,N,T}) where {F<:PairForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(dest)
    size(dest, 1) == size(dest, 2) || throw(ArgumentError("cannot copy a SymplecticGauss object to a non-square matrix."))
    n, k, c, d = (src.form).n, src.k, src.c, src.d
    fill!(dest, zero(T))
    @inbounds for i in axes(dest, 1)
        dest[i, i] = oneunit(T)
    end
    dest[2k-3, 2k-3] = c
    dest[2k-1, 2k-1] = c
    dest[2k-3, 2k] = d
    dest[2k-1, 2k-2] = d
    dest[2k-2, 2k-2] = inv(c)
    dest[2k, 2k] = inv(c)
    return dest
end
@inline function Base.copyto!(dest::Symplectic, src::SymplecticGauss{F,N,T}) where {F<:PairForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(dest)
    @assert dest.form == src.form
    size(dest, 1) == size(dest, 2) || throw(ArgumentError("cannot copy a SymplecticGauss object to a non-square matrix."))
    n, k, c, d = (src.form).n, src.k, src.c, src.d
    fill!(dest, zero(T))
    @inbounds for i in axes(dest, 1)
        dest[i, i] = oneunit(T)
    end
    dest[2k-3, 2k-3] = c
    dest[2k-1, 2k-1] = c
    dest[2k-3, 2k] = d
    dest[2k-1, 2k-2] = d
    dest[2k-2, 2k-2] = inv(c)
    dest[2k, 2k] = inv(c)
    return dest
end

@inline function LinearAlgebra.lmul!(L::SymplecticGauss{F,N,T}, A::AbstractMatrix) where {F<:BlockForm,N<:Int,T}
    n, k, c, d = L.form.n, L.k, L.c, L.d
    @inbounds for j in axes(A, 2)
        a_km1 = A[k-1, j]
        a_k   = A[k, j]
        a_nkm1 = A[n+k-1, j]
        a_nk   = A[n+k, j]
        A[k-1, j] = c * a_km1 + d * a_nk
        A[k, j] = c * a_k + d * a_nkm1
        A[n+k-1, j] = inv(c) * a_nkm1
        A[n+k, j] = inv(c) * a_nk
    end
    return A
end
@inline function LinearAlgebra.lmul!(x::SymplecticGauss{F,N,T}, y::AbstractVector) where {F<:BlockForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(y)
    n, k, c, d = x.form.n, x.k, x.c, x.d
    a_km1 = y[k-1]
    a_k   = y[k]
    a_nkm1 = y[n+k-1]
    a_nk   = y[n+k]
    
    y[k-1] = c * a_km1 + d * a_nk
    y[k] = c * a_k + d * a_nkm1
    y[n+k-1] = inv(c) * a_nkm1
    y[n+k] = inv(c) * a_nk
    return y
end
@inline function LinearAlgebra.rmul!(A::AbstractMatrix, L::SymplecticGauss{F,N,T}) where {F<:BlockForm,N<:Int,T}
    n, k, c, d = L.form.n, L.k, L.c, L.d
    @inbounds for i in axes(A, 1)
        a_km1 = A[i, k-1]
        a_k   = A[i, k]
        a_nkm1 = A[i, n+k-1]
        a_nk   = A[i, n+k]
        
        A[i, k-1] = a_km1 * c
        A[i, k] = a_k * c
        A[i, n+k-1] = a_k * d + a_nkm1 * inv(c)
        A[i, n+k] = a_km1 * d + a_nk * inv(c)
    end
    return A
end
@inline function LinearAlgebra.lmul!(L::SymplecticGauss{F,N,T}, Y::AbstractMatrix) where {F<:PairForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(Y)
    size(Y, 1) == size(Y, 2) || throw(ArgumentError("cannot compute the matrix product between a SymplecticGauss object and a non-square matrix."))
    n, k, c, d = L.form.n, L.k, L.c, L.d
    @inbounds for j in axes(Y, 2)
        y_q1 = Y[2k-3, j]
        y_p1 = Y[2k-2, j]
        y_q2 = Y[2k-1, j]
        y_p2 = Y[2k, j]
        
        Y[2k-3, j] = c * y_q1 + d * y_p2
        Y[2k-1, j] = c * y_q2 + d * y_p1
        Y[2k-2, j] = inv(c) * y_p1
        Y[2k, j] = inv(c) * y_p2
    end
    return Y
end
@inline function LinearAlgebra.lmul!(x::SymplecticGauss{F,N,T}, y::AbstractVector) where {F<:PairForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(y)
    n, k, c, d = x.form.n, x.k, x.c, x.d
    y_q1 = y[2k-3]
    y_p1 = y[2k-2]
    y_q2 = y[2k-1]
    y_p2 = y[2k]
    
    y[2k-3] = c * y_q1 + d * y_p2
    y[2k-1] = c * y_q2 + d * y_p1
    y[2k-2] = inv(c) * y_p1
    y[2k] = inv(c) * y_p2
    return y
end
@inline function LinearAlgebra.rmul!(X::AbstractMatrix, L::SymplecticGauss{F,N,T}) where {F<:PairForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(X)
    size(X, 1) == size(X, 2) || throw(ArgumentError("cannot compute the matrix product between a SymplecticGauss object and a non-square matrix."))
    n, k, c, d = L.form.n, L.k, L.c, L.d
    @inbounds for i in axes(X, 1)
        x_q1 = X[i, 2k-3]
        x_p1 = X[i, 2k-2]
        x_q2 = X[i, 2k-1]
        x_p2 = X[i, 2k]
        
        X[i, 2k-3] = x_q1 * c
        X[i, 2k-1] = x_q2 * c
        X[i, 2k-2] = x_q2 * d + x_p1 * inv(c)
        X[i, 2k] = x_q1 * d + x_p2 * inv(c)
    end
    return X
end

Base.:(*)(x::SymplecticGauss, y::SymplecticGauss) = x.k == y.k ? SymplecticGauss(x.form, x.k, x.c * y.c, x.c * y.d + x.d * inv(y.c)) : Symplectic(x.form, Matrix(x) * Matrix(y))
Base.:(*)(x::SymplecticGauss, y::Symplectic) = x.form == y.form ? Symplectic(x.form, x * y.data) : x * y.data
Base.:(*)(x::Symplectic, y::SymplecticGauss) = x.form == y.form ? Symplectic(x.form, x.data * y) : x.data * y
Base.:(/)(x::SymplecticGauss, y::SymplecticGauss) = x.k == y.k ? x * inv(y) : Symplectic(x.form, Matrix(x) / Matrix(y))
Base.:(/)(x::SymplecticGauss, y::Symplectic) = x.form == y.form ? Symplectic(x.form, x * inv(y.data)) : x * inv(y.data)
Base.:(/)(x::Symplectic, y::SymplecticGauss) = x.form == y.form ? Symplectic(x.form, x.data * inv(y)) : x.data * inv(y)
Base.:(/)(x::SymplecticGauss, y::AbstractMatrix) = x * inv(y)
Base.:(/)(x::AbstractMatrix, y::SymplecticGauss) = x * inv(y)
Base.:(\)(x::SymplecticGauss, y::SymplecticGauss) = x.k == y.k ? inv(x) * y : Symplectic(x.form, Matrix(x) \ Matrix(y))
Base.:(\)(x::SymplecticGauss, y::Symplectic) = x.form == y.form ? Symplectic(x.form, inv(x) * y.data) : inv(x) * y.data
Base.:(\)(x::Symplectic, y::SymplecticGauss) = x.form == y.form ? Symplectic(x.form, inv(x.data) * y) : inv(x.data) * y
Base.:(\)(x::SymplecticGauss, y::AbstractMatrix) = inv(x) * y
Base.:(\)(x::AbstractMatrix, y::SymplecticGauss) = inv(x) * y

function Base.replace_in_print_matrix(G::SymplecticGauss, i::Integer, j::Integer, s::AbstractString)
    if G[i,j] == 0
        return Base.replace_with_centered_mark(s)
    else
        return s
    end
end