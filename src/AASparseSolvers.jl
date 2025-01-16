module AASparseSolvers
using LinearAlgebra
include("wrappers.jl")

mutable struct AASparseMatrix{T<:vTypes}# <: AbstractMatrix{T}
    matrix::SparseMatrix{T}
    _colptr::Vector{Clong}
    _rowval::Vector{Cint}
    _nzval::Vector{T}  
end

# I use StridedVector here because it allows for views/references,
# so you can do shallow copies: same pointed-to data. Better way?  
function AASparseMatrix(n::Int, m::Int,
            col::StridedVector{Clong}, row::StridedVector{Cint},
            data::StridedVector{T}) where T<:vTypes
    @assert stride(col, 1) == 1 && stride(row, 1) == 1 && stride(data, 1) == 1
    # I'm assuming here that pointer(row) == pointer(_row_inds),
    # ie that col, row, and data are passed by reference, not by value.
    s = SparseMatrixStructure(n, m, pointer(col),
                    pointer(row), ATT_ORDINARY, 1)
    m = SparseMatrix(s, pointer(data))
    return AASparseMatrix(m, col, row, data)
end

function AASparseMatrix(sparseM::SparseMatrixCSC{T, Int64}) where T<:vTypes
    c = Clong.(sparseM.colptr .+ -1)
    r = Cint.(sparseM.rowval .+ -1)
    vals = copy(sparseM.nzval)
    return AASparseMatrix(size(sparseM)..., c, r, vals)
end

Base.size(M::AASparseMatrix) = (M.matrix.structure.rowCount, M.matrix.structure.columnCount)
Base.eltype(M::AASparseMatrix) = eltype(M._nzval)

function Base.getindex(M::AASparseMatrix, i::Int, j::Int)
    @assert all((1, 1) .<= (i,j) .<= size(M))
    (startCol, endCol) = (M._colptr[j], M._colptr[j+1]-1) .+ 1
    rowsInCol = @view M._rowval[startCol:endCol]
    ind = searchsortedfirst(rowsInCol, i-1)
    if ind <= length(rowsInCol) && rowsInCol[ind] == i-1
        return M._nzval[startCol+ind-1]
    end
    return zero(eltype(M))
end

# Creates a new structure, referring to the same data,
# but with the transpose flag (in attributes) flipped.
# TODO: untested
Base.transpose(M::AASparseMatrix) = AASparseMatrix(SparseGetTranspose(M.matrix),
                        M._colptr, M._rowval, M._nzval)

function Base.:(*)(A::AASparseMatrix{T}, x::StridedVecOrMat{T}) where T<:vTypes
    @assert size(x)[1] == size(A)[2]
    y = Array{T}(undef, size(A)[1], size(x)[2:end]...)
    SparseMultiply(A.matrix, x, y)
    return y
end

function Base.:(*)(alpha::T, A::AASparseMatrix{T},
                            x::StridedVecOrMat{T}) where T<:vTypes
    @assert size(x)[1] == size(A)[2]
    y = Array{T}(undef, size(A)[1], size(x)[2:end]...)
    SparseMultiply(alpha, A.matrix, x, y)
    return y
end

# modifies its LAST argument.
function muladd!(A::AASparseMatrix{T}, x::StridedVecOrMat{T},
                    y::StridedVecOrMat{T}) where T<:vTypes
    @assert size(x) == size(y) && size(x)[1] == size(A)[2]
    SparseMultiplyAdd(A.matrix, x, y)
end

# modifies its LAST argument.
function muladd!(alpha::T, A::AASparseMatrix{T},
                x::StridedVecOrMat{T}, y::StridedVecOrMat{T}) where T<:vTypes
    @assert size(x) == size(y) && size(x)[1] == size(A)[2]
    SparseMultiplyAdd(alpha, A.matrix, x, y)
end

# I should probably make this a subclass of Factorization.
mutable struct AAFactorization{T<:vTypes} <: LinearAlgebra.Factorization{T}
    aa_matrix::SparseMatrix{T}
    # these don't *quite* match SparseArrayCSC fields: 1 vs 0 indexing,
    # Cint vs Clong for row indices.
    _col_inds::Vector{Clong}
    _row_inds::Vector{Cint}
    _data::Vector{T}
    _factorization::Union{Nothing, SparseOpaqueFactorization}
end

# if symmetric, only have the one triangle (upper or lower) populated.
# (if both are populated, things work still: it's just redundant.)
# TODO: way that's compatible with julia's special matrix classes?
function AAFactorization(sparseM::SparseMatrixCSC{T, Int64},
                                symmetric::Bool = false,
                                upperTriangle::Bool = false) where T<:vTypes
    kind = nothing
    if symmetric && upperTriangle
        kind = AASparseSolvers.ATT_SYMMETRIC | AASparseSolvers.ATT_UPPER_TRIANGLE 
    elseif symmetric
        kind = AASparseSolvers.ATT_SYMMETRIC | AASparseSolvers.ATT_LOWER_TRIANGLE
    else
        kind = AASparseSolvers.ATT_ORDINARY
    end
    c = Clong.(sparseM.colptr .+ -1)
    r = Cint.(sparseM.rowval .+ -1)
    vals = copy(sparseM.nzval)
    s = SparseMatrixStructure(size(sparseM)..., pointer(c), pointer(r), kind, 1)
    obj = AAFactorization(
        SparseMatrix(s, pointer(vals)),
        c, r, vals,
        nothing
    )
    function cleanup(aa_fact)
        if !isnothing(aa_fact._factorization)
            AASparseSolvers.SparseCleanup(aa_fact._factorization)
        end
    end
    return finalizer(cleanup, obj)
end

# TODO: I could make this follow the defaults and naming conventions of
# https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.factorize
function factor!(aa_fact::AAFactorization{T},
                kind::Union{SparseFactorization_t, Nothing} = nothing) where T<:vTypes
    if isnothing(aa_fact._factorization)
        if isnothing(kind)
            ordinary = aa_fact.aa_matrix.structure.attributes == AASparseSolvers.ATT_ORDINARY
            kind = ordinary ?  AASparseSolvers.SparseFactorizationQR :
                                    AASparseSolvers.SparseFactorizationLDLT
        end
        aa_fact._factorization = AASparseSolvers.SparseFactor(kind, aa_fact.aa_matrix, true)
    end
end

function solve(aa_fact::AAFactorization{T}, b::StridedVecOrMat{T}) where T<:vTypes
    @assert aa_fact.aa_matrix.structure.columnCount == size(b, 1)
    factor!(aa_fact)
    x = Array{T}(undef, aa_fact.aa_matrix.structure.columnCount, size(b)[2:end]...)
    AASparseSolvers.SparseSolve(aa_fact._factorization, b, x)
    return x
end

function solve!(aa_fact::AAFactorization{T}, xb::StridedVecOrMat{T}) where T<:vTypes
    @assert (xb isa StridedVector) ||
            (aa_fact.aa_matrix.structure.rowCount == 
            aa_fact.aa_matrix.structure.columnCount) "Can't in-place solve:" *
            " x and b are different sizes and Julia cannot resize a matrix."
    factor!(aa_fact)
    AASparseSolvers.SparseSolve(aa_fact._factorization, xb)
    return xb # because KLU also returns
end

LinearAlgebra.ldiv!(aa_fact::AAFactorization{T}, xb::StridedVecOrMat{T}) where T<:vTypes =
        solve!(aa_fact, xb)

function LinearAlgebra.ldiv!(x::StridedVecOrMat{T},
                            aa_fact::AAFactorization{T},
                            b::StridedVecOrMat{T}) where T<:vTypes
    @assert aa_fact.aa_matrix.structure.columnCount == size(b, 1)
    factor!(aa_fact)
    AASparseSolvers.SparseSolve(aa_fact._factorization, b, x)
    return x
end

export AAFactorization, solve, solve!, AASparseMatrix, muladd!

end # module AASparseSolvers
