struct SymplecticHouseholder{F<:SymplecticForm,N<:Int,T} <: AbstractMatrix{T}
    form::F
    k::N
    P::T
    SymplecticHouseholder(form::F, k::N, P::T) where {F<:SymplecticForm,N<:Int,T} = new{F,N,T}(form, k, P)
end
SymplecticHouseholder(x::SymplecticHouseholder) = x

function householder(form::SymplecticForm, k::Int, v::V) where {V}
    k < form.n || throw(ArgumentError("k must be less than n."))
    length(v) == (form.n - k + 1) || throw(ArgumentError("the length of v must be equal to n - k + 1"))
    vt = transpose(v)
    P = I - (2 / (vt * v)) * v * vt
    return SymplecticHouseholder(form, k, P)
end

function Base.Matrix(x::SymplecticHouseholder{F,N,T}) where {F<:BlockForm,N<:Int,T}
    n, k, P = (x.form).n, x.k, x.P
    M = Matrix{eltype(P)}(I, 2n, 2n)
    Base.copyto!(@view(M[k:n,k:n]), P)
    Base.copyto!(@view(M[n+k:2n,n+k:2n]), P)
    return M
end
function Base.Matrix(x::SymplecticHouseholder{F,N,T}) where {F<:PairForm,N<:Int,T}
    n, k, P = (x.form).n, x.k, x.P
    M = Matrix{eltype(P)}(I, 2n, 2n)
    @inbounds for i in Base.OneTo(n-k+1)
        @inbounds for j in Base.OneTo(n-k+1)
            M[2(i+k)-3, 2(j+k)-3] = P[i,j]
            M[2(i+k)-2, 2(j+k)-2] = P[i,j]
        end
    end
    return M
end
Base.Array(x::SymplecticHouseholder) = Matrix(x)
function Base.AbstractMatrix{T1}(x::SymplecticHouseholder{F,N,T2}) where {F<:BlockForm,N<:Int,T1,T2}
    n, k, P = (x.form).n, x.k, x.P
    M = Matrix{eltype(T1)}(I, 2n, 2n)
    Base.copyto!(@view(M[k:n,k:n]), P)
    Base.copyto!(@view(M[n+k:2n,n+k:2n]), P)
    return M
end
function Base.AbstractMatrix{T1}(x::SymplecticHouseholder{F,N,T2}) where {F<:PairForm,N<:Int,T1,T2}
    n, k, P = (x.form).n, x.k, x.P
    M = Matrix{eltype(P)}(I, 2n, 2n)
    @inbounds for i in Base.OneTo(n-k+1)
        @inbounds for j in Base.OneTo(n-k+1)
            M[2(i+k)-3, 2(j+k)-3] = P[i,j]
            M[2(i+k)-2, 2(j+k)-2] = P[i,j]
        end
    end
    return M
end

Base.checkbounds(x::SymplecticHouseholder, i::Int, j::Int) = (form = x.form; i <= 2 * form.n && j <= 2 * form.n)
Base.checkbounds(x::SymplecticHouseholder, i::Int) = (form = x.form; i <= (2 * form.n)^2)
Base.isequal(x::SymplecticHouseholder, y::SymplecticHouseholder) = x.form == y.form && x.k == y.k && x.P == y.P
Base.isapprox(x::SymplecticHouseholder, y::SymplecticHouseholder) = x.form == y.form && isapprox(x.k, y.k) && isapprox(x.P, y.P)
Base.size(x::SymplecticHouseholder) = (form = x.form; (2 * form.n, 2 * form.n))
Base.size(x::SymplecticHouseholder, dim::Int) = (form = x.form; 2 * form.n)
Base.length(x::SymplecticHouseholder) = prod(size(x))
Base.axes(x::SymplecticHouseholder) = (form = x.form; (Base.OneTo(2 * form.n), Base.OneTo(2 * form.n)))
Base.eltype(x::SymplecticHouseholder) = eltype(x.P)

Base.@propagate_inbounds function Base.getindex(x::SymplecticHouseholder{F,N,T}, i::Int, j::Int) where {F<:BlockForm,N<:Int,T}
    @boundscheck checkbounds(x, i, j)
    n, k, P = (x.form).n, x.k, x.P
    if i <= n && j <= n
        if i < k || j < k
            return i == j ? oneunit(eltype(P)) : zero(eltype(P))
        else
            return P[i - k + 1, j - k + 1]
        end
    elseif i > n && j > n
        if i < n + k || j < n + k
            return i == j ? oneunit(eltype(P)) : zero(eltype(P))
        else
            return P[i - n - k + 1, j - n - k + 1]
        end
    else
        return zero(eltype(P))
    end
end
Base.@propagate_inbounds function Base.getindex(x::SymplecticHouseholder{F,N,T}, i::Int, j::Int) where {F<:PairForm,N<:Int,T}
    @boundscheck checkbounds(x, i, j)
    n, k, P = (x.form).n, x.k, x.P
    if i < 2k-1
        if i == j
            return oneunit(eltype(P))
        else
            return zero(eltype(P))
        end
    elseif 2k-1 <= i && 2k-1 <= j
        if i % 2 == j % 2
            return P[cld(i,2) - k + 1, cld(j,2) - k + 1]
        else
            return zero(eltype(P))
        end
    else 
        return zero(eltype(P))
    end
end
Base.@propagate_inbounds function Base.getindex(x::SymplecticHouseholder, ind::Int)
    @boundscheck checkbounds(x, ind)
    n2 = 2 * x.form.n
    c, r = divrem(ind - 1, n2)
    return x[r + 1, c + 1]
end

LinearAlgebra.adjoint(x::SymplecticHouseholder) = x
LinearAlgebra.inv(x::SymplecticHouseholder) = x
Base.copy(x::SymplecticHouseholder) = SymplecticHouseholder(x.form, copy(x.k), copy(x.P))

@inline function Base.copyto!(dest::AbstractMatrix, src::SymplecticHouseholder{F,N,T}) where {F<:BlockForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(dest)
    size(dest, 1) == size(dest, 2) || throw(ArgumentError("cannot copy a SymplecticHouseholder object to a non-square matrix."))
    n, k, P = (src.form).n, src.k, src.P
    fill!(dest, zero(eltype(P)))
    @inbounds for i in axes(dest, 1)
        dest[i, i] = oneunit(eltype(P))
    end
    Base.copyto!(@view(dest[k:n, k:n]), P)
    Base.copyto!(@view(dest[n+k:2n, n+k:2n]), P)
    return dest
end
@inline function Base.copyto!(dest::Symplectic, src::SymplecticHouseholder{F,N,T}) where {F<:BlockForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(dest)
    @assert dest.form == src.form
    size(dest, 1) == size(dest, 2) || throw(ArgumentError("cannot copy a SymplecticHouseholder object to a non-square matrix."))
    n, k, P = (src.form).n, src.k, src.P
    fill!(dest, zero(eltype(P)))
    @inbounds for i in axes(dest, 1)
        dest[i, i] = oneunit(eltype(P))
    end
    Base.copyto!(@view(dest[k:n, k:n]), P)
    Base.copyto!(@view(dest[n+k:2n, n+k:2n]), P)
    return dest
end
@inline function Base.copyto!(dest::AbstractMatrix, src::SymplecticHouseholder{F,N,T}) where {F<:PairForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(dest)
    size(dest, 1) == size(dest, 2) || throw(ArgumentError("cannot copy a SymplecticHouseholder object to a non-square matrix."))
    n, k, P = (src.form).n, src.k, src.P
    fill!(dest, zero(eltype(P)))
    @inbounds for i in axes(dest, 1)
        dest[i, i] = oneunit(eltype(P))
    end
    @inbounds for i in Base.OneTo(n-k+1)
        @inbounds for j in Base.OneTo(n-k+1)
            dest[2(i+k)-3, 2(j+k)-3] = P[i,j]
            dest[2(i+k)-2, 2(j+k)-2] = P[i,j]
        end
    end
    return dest
end
@inline function Base.copyto!(dest::Symplectic, src::SymplecticHouseholder{F,N,T}) where {F<:PairForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(dest)
    @assert dest.form == src.form
    size(dest, 1) == size(dest, 2) || throw(ArgumentError("cannot copy a SymplecticHouseholder object to a non-square matrix."))
    n, k, P = (src.form).n, src.k, src.P
    fill!(dest, zero(eltype(P)))
    @inbounds for i in axes(dest, 1)
        dest[i, i] = oneunit(eltype(P))
    end
    @inbounds for i in Base.OneTo(n-k+1)
        @inbounds for j in Base.OneTo(n-k+1)
            dest[2(i+k)-3, 2(j+k)-3] = P[i,j]
            dest[2(i+k)-2, 2(j+k)-2] = P[i,j]
        end
    end
    return dest
end

@inline function LinearAlgebra.lmul!(H::SymplecticHouseholder{F,N,T}, A::AbstractMatrix) where {F<:BlockForm,N<:Int,T}
    n, k, P = H.form.n, H.k, H.P
    m = n - k + 1
    temp = Vector{eltype(A)}(undef, m)
    @inbounds for j in axes(A, 2)
        @inbounds for i in 1:m
            temp[i] = A[k + i - 1, j]
        end
        mul!(view(A, k:n, j), P, temp)
        @inbounds for i in 1:m
            temp[i] = A[n + k + i - 1, j]
        end
        mul!(view(A, n+k:2n, j), P, temp)
    end
    return A
end
@inline function LinearAlgebra.lmul!(x::SymplecticHouseholder{F,N,T}, y::AbstractVector) where {F<:BlockForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(y)
    n, k, P = x.form.n, x.k, x.P
    @views begin
        ytop = view(y, k:n)
        ybot = view(y, n+k:2n)
        mul!(ytop, P, ytop)
        mul!(ybot, P, ybot)
    end
    return y
end
@inline function LinearAlgebra.rmul!(A::AbstractMatrix, H::SymplecticHouseholder{F,N,T}) where {F<:BlockForm,N<:Int,T}
    n, k, P = H.form.n, H.k, H.P
    m = n - k + 1
    temp = Vector{eltype(A)}(undef, m)
    @inbounds for i in axes(A, 1)
        @inbounds for j in 1:m
            temp[j] = A[i, k + j - 1]
        end
        mul!(view(A, i, k:n), P', temp)
        
        @inbounds for j in 1:m
            temp[j] = A[i, n + k + j - 1]
        end
        mul!(view(A, i, n+k:2n), P', temp)
    end
    return A
end
@inline function LinearAlgebra.lmul!(x::SymplecticHouseholder{F,N,T}, y::AbstractMatrix) where {F<:PairForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(y)
    size(y, 1) == size(y, 2) || throw(ArgumentError("cannot compute the matrix product between a SymplecticHouseholder object and a non-square matrix."))
    n, k, P = x.form.n, x.k, x.P
    @inbounds for i in Base.OneTo(n-k+1)
        @views yq = y[2(i+k-1)-1, :]
        @views yp = y[2(i+k-1), :]
        tq, tp = zero(yq), tp = zero(yp)
        @inbounds for j in Base.OneTo(n-k+1)
            @views yjq = y[2(j+k-1)-1, :]
            @views yjp = y[2(j+k-1), :]
            tq .+= P[i,j] .* yjq
            tp .+= P[i,j] .* yjp
        end
        yq .= tq
        yp .= tp
    end
    return y
end
@inline function LinearAlgebra.lmul!(x::SymplecticHouseholder{F,N,T}, y::AbstractVector) where {F<:PairForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(y)
    n, k, P = x.form.n, x.k, x.P
    @inbounds for i in Base.OneTo(n-k+1)
        tq, tp = zero(eltype(y)), zero(eltype(y))
        for j in Base.OneTo(n - k + 1)
            tq += P[i,j] * y[2(j+k-1)-1]
            tp += P[i,j] * y[2(j+k-1)]
        end
        y[2(i+k-1)-1] = tq
        y[2(i+k-1)] = tp
    end
    return y
end
@inline function LinearAlgebra.rmul!(x::AbstractMatrix, y::SymplecticHouseholder{F,N,T}) where {F<:PairForm,N<:Int,T}
    LinearAlgebra.require_one_based_indexing(x)
    size(x, 1) == size(x, 2) || throw(ArgumentError("cannot compute the matrix product between a SymplecticHouseholder object and a non-square matrix."))
    n, k, P = y.form.n, y.k, y.P
    @inbounds for j in Base.OneTo(n - k + 1)
        @views xq = x[:, 2(j+k-1)-1]
        @views xp = x[:, 2(j+k-1)]
        tq, tp = zero(xq), zero(xp)
        @inbounds for i in Base.OneTo(n-k+1)
            @views xiq = x[:, 2(i+k-1)-1]
            @views xip = x[:, 2(i+k-1)]
            tq .+= xiq .* P[i,j]
            tp .+= xip .* P[i,j]
        end
        xq .= tq
        xp .= tp
    end
    return x
end
Base.:(*)(x::SymplecticHouseholder, y::SymplecticHouseholder) = x.k == y.k ? SymplecticHouseholder(x.form, x.k, x.P * y.P) : Symplectic(x.form, Matrix(x) * Matrix(y))
Base.:(*)(x::SymplecticHouseholder, y::Symplectic) = x.form == y.form ? Symplectic(x.form, x * y.data) : x * y.data
Base.:(*)(x::Symplectic, y::SymplecticHouseholder) = x.form == y.form ? Symplectic(x.form, x.data * y) : x.data * y
Base.:(/)(x::SymplecticHouseholder, y::SymplecticHouseholder) = x.k == y.k ? SymplecticHouseholder(x.form, x.k, x.P / y.P) : Symplectic(x.form, Matrix(x) / Matrix(y))
Base.:(/)(x::SymplecticHouseholder, y::Symplectic) = x.form == y.form ? Symplectic(x.form, x * inv(y.data)) : x * inv(y.data)
Base.:(/)(x::Symplectic, y::SymplecticHouseholder) = x.form == y.form ? Symplectic(x.form, x.data * inv(y)) : x.data * inv(y)
Base.:(/)(x::SymplecticHouseholder, y::AbstractMatrix) = x * inv(y)
Base.:(/)(x::AbstractMatrix, y::SymplecticHouseholder) = x * inv(y)
Base.:(\)(x::SymplecticHouseholder, y::SymplecticHouseholder) = x.k == y.k ? SymplecticHouseholder(x.form, x.k, x.P \ y.P) : Symplectic(x.form, Matrix(x) \ Matrix(y))
Base.:(\)(x::SymplecticHouseholder, y::Symplectic) = x.form == y.form ? Symplectic(x.form, inv(x) * y.data) : inv(x) * y.data
Base.:(\)(x::Symplectic, y::SymplecticHouseholder) = x.form == y.form ? Symplectic(x.form, inv(x.data) * y) : inv(x.data) * y
Base.:(\)(x::SymplecticHouseholder, y::AbstractMatrix) = inv(x) * y
Base.:(\)(x::AbstractMatrix, y::SymplecticHouseholder) = inv(x) * y

function Base.replace_in_print_matrix(x::SymplecticHouseholder{F,N,T}, i::Integer, j::Integer, s::AbstractString) where {F<:BlockForm,N<:Int,T}
    n, k = x.form.n, x.k
    in_upper_block = (k <= i <= n) && (k <= j <= n)
    in_lower_block = (n + k <= i <= 2n) && (n + k <= j <= 2n)
    is_diag = (i == j)
    if in_upper_block || in_lower_block || is_diag
        return s
    else
        return Base.replace_with_centered_mark(s)
    end
end
function Base.replace_in_print_matrix(x::SymplecticHouseholder{F,N,T}, i::Integer, j::Integer, s::AbstractString) where {F<:PairForm,N<:Int,T}
    n, k = x.form.n, x.k
    in_interleaved_block = (2k - 1 <= i <= 2n) && (2k - 1 <= j <= 2n) && (i % 2 == j % 2)
    is_diag = (i == j)
    if in_interleaved_block || is_diag
        return s
    else
        return Base.replace_with_centered_mark(s)
    end
end