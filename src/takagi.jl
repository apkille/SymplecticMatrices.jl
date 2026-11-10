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

function takagi(x::AbstractMatrix{T}) where {T<:Union{Real,Complex}}
    fact = svd(Symmetric(x))
    Tt = T <: Complex ? T : ComplexF64

    #Custom computation of the square root to avoid jumps from the principal branch

    #Schur decompose of the matrix inside the square root
    #This matrix is normal, so its T matrix is diagonal
    sF = schur(conj(transpose(fact.U) * fact.V))  
    sF_eivals = diag(sF.T)

    #Find the unique angles, mod 2 * pi, of the eigenvalues and sort them
    sF_angles = angle.(sF_eivals)
    period = oftype(sF_angles[1], 2 * pi)
    sF_sorted_angles = sort(unique(mod.(sF_angles, period)))

    #Find the difference between these angles, including the difference between the first and last angle (mod 2 * pi)
    #Select the largest difference and find the midpoint of the corresponding arc
    sF_angles_diff = diff(vcat(sF_sorted_angles, sF_sorted_angles[1] + period))
    indx_max_diff = argmax(sF_angles_diff)
    mid_point = sF_sorted_angles[indx_max_diff] + 0.5 * sF_angles_diff[indx_max_diff]
    
    #Wrap the midpoint around 2 * pi and then shift it by -pi
    shift = mod(-mid_point, period) - pi

    #Rotate all the eigenvalues so that the midpoint of the largest arc lies at +- pi.
    #The other angles will be far from the branch cut of the complex square root
    shifted_sF_eivals = sF_eivals .* exp.(im * shift)

    #Compute the square root of the shifted eigenvalues, reconstruct a shifted square root
    sqrt_shifted_sF_eivals = sqrt.(shifted_sF_eivals)
    sqrt_shifted = sF.Z * Diagonal(sqrt_shifted_sF_eivals) * sF.Z'

    #Undo the rotation
    Q = zeros(Tt, size(x))
    mul!(Q, fact.U, sqrt_shifted .* exp.(-im * shift / 2))

    return Takagi{Tt}(Q, fact.S)
end
function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::Takagi{<:Any,<:AbstractArray,<:AbstractVector})
    Base.summary(io, F); println(io)
    println(io, "Q factor:")
    Base.show(io, mime, F.Q)
    println(io, "\nsingular values:")
    Base.show(io, mime, F.S)
end