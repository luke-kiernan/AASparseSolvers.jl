module AASparseSolvers
using LinearAlgebra
include("wrappers.jl")

@doc """Matrix wrapper, containing the Apple sparse matrix struct
and the pointed-to data. Construct from a `SparseMatrixCSC`.

Multiplication (`*`) and multiply-add (`muladd!`) with both
`Vector` and `Matrix` objects are working.
`transpose` creates a new matrix structure with the opposite
transpose flag, that references the same CSC data.
"""
mutable struct AASparseMatrix{T<:vTypes} <: AbstractMatrix{T}
    matrix::SparseMatrix{T}
    _colptr::Vector{Clong}
    _rowval::Vector{Cint}
    _nzval::Vector{T}
end

# I use StridedVector here because it allows for views/references,
# so you can do shallow copies: same pointed-to data. Better way?
"""Constructor for advanced usage: col and row here are 0-indexed CSC data.
Could allow for shared  `_colptr`, `_rowval`, `_nzval` between multiple
structs via views or references. Currently unused."""
function AASparseMatrix(n::Int, m::Int,
            col::StridedVector{Clong}, row::StridedVector{Cint},
            data::StridedVector{T},
            attributes::att_type = ATT_ORDINARY) where T<:vTypes
    @assert stride(col, 1) == 1 && stride(row, 1) == 1 && stride(data, 1) == 1
    # I'm assuming here that pointer(row) == pointer(_row_inds),
    # ie that col, row, and data are passed by reference, not by value.
    s = SparseMatrixStructure(n, m, pointer(col),
                    pointer(row), attributes, 1)
    m = SparseMatrix(s, pointer(data))
    return AASparseMatrix(m, col, row, data)
end

function AASparseMatrix(sparseM::SparseMatrixCSC{T, Int64},
                        attributes::att_type = ATT_ORDINARY) where T<:vTypes
    if issymmetric(sparseM) && attributes == ATT_ORDINARY
        return AASparseMatrix(tril(sparseM), ATT_SYMMETRIC | ATT_LOWER_TRIANGLE)
    elseif (istril(sparseM) || istriu(sparseM)) && attributes == ATT_ORDINARY
        attributes = istril(sparseM) ? ATT_TRI_LOWER : ATT_TRI_UPPER
    end
    if attributes in (ATT_TRI_LOWER, ATT_TRI_UPPER) &&
                    all(diag(sparseM) .== one(eltype(sparseM)))
        attributes |= ATT_UNIT_TRIANGULAR
    end
    c = Clong.(sparseM.colptr .+ -1)
    r = Cint.(sparseM.rowval .+ -1)
    vals = copy(sparseM.nzval)
    return AASparseMatrix(size(sparseM)..., c, r, vals, attributes)
end

Base.size(M::AASparseMatrix) = (M.matrix.structure.rowCount,
                                    M.matrix.structure.columnCount)
Base.eltype(M::AASparseMatrix) = eltype(M._nzval)
LinearAlgebra.issymmetric(M::AASparseMatrix) = (M.matrix.structure.attributes &
                                                ATT_KIND_MASK) == ATT_SYMMETRIC
LinearAlgebra.istriu(M::AASparseMatrix) = (M.matrix.structure.attributes &
                            ATT_TRI_LOWER) == ATT_TRI_LOWER
LinearAlgebra.istril(M::AASparseMatrix) = (M.matrix.structure.attributes &
                            ATT_TRI_UPPER) == ATT_TRI_UPPER

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

function Base.getindex(M::AASparseMatrix, i::Int)
    @assert 1 <= i <= size(M)[1]*size(M)[2]
    return M[(i-1) % size(M)[1] + 1, div(i-1, size(M)[1]) + 1]
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

"""
Computes y += A*x in place. Note that this modifies its LAST argument.
"""
function muladd!(A::AASparseMatrix{T}, x::StridedVecOrMat{T},
                    y::StridedVecOrMat{T}) where T<:vTypes
    @assert size(x) == size(y) && size(x)[1] == size(A)[2]
    SparseMultiplyAdd(A.matrix, x, y)
end

"""
Computes y += alpha*A*x in place. Note that this modifies its LAST argument.
"""
function muladd!(alpha::T, A::AASparseMatrix{T},
                x::StridedVecOrMat{T}, y::StridedVecOrMat{T}) where T<:vTypes
    @assert size(x) == size(y) && size(x)[1] == size(A)[2]
    SparseMultiplyAdd(alpha, A.matrix, x, y)
end

factorize(A::AASparseMatrix{T}) where T<:vTypes = AAFactorization(A)

@doc """Factorization object.

Create via `f = AAFactorization(A::SparseMatrixCSC{T, Int64})`. Calls to `solve`,
`ldiv`, and their in-place versions require explicitly passing in the
factorization object as the first argument. On construction, the struct stores a
placeholder yet-to-be-factored object: the factorization is computed upon the first call
to `solve`, or by explicitly calling `factor!`. If the matrix is symmetric, it defaults to
a Cholesky factorization; otherwise, it defaults to QR.
"""
mutable struct AAFactorization{T<:vTypes} <: LinearAlgebra.Factorization{T}
    matrixObj::AASparseMatrix{T}
    _factorization::SparseOpaqueFactorization{T}
end

# returns an AAFactorization containing A and a dummy "yet-to-be-factored" factorization.
function AAFactorization(A::AASparseMatrix{T}) where T<:vTypes
    obj = AAFactorization(A, SparseOpaqueFactorization(T))
    function cleanup(aa_fact)
        # If it's yet-to-be-factored, then there's nothing to release
        if !(aa_fact._factorization.status in (SparseYetToBeFactored,
                                                            SparseStatusReleased))
            SparseCleanup(aa_fact._factorization)
        end
    end
    return finalizer(cleanup, obj)
end

# julia's LinearAlgebra module doesn't provide similar constructors.
AAFactorization(M::SparseMatrixCSC{T, Int64}) where T<:vTypes =
                                        AAFactorization(AASparseMatrix(M))

# easiest way to make this follow the defaults and naming conventions of LinearAlgebra?
# TODO: add tests for the different kinds of factorizations, beyond QR.
function factor!(aa_fact::AAFactorization{T},
            kind::SparseFactorization_t = SparseFactorizationTBD) where T<:vTypes
    if aa_fact._factorization.status == SparseYetToBeFactored
        if kind == SparseFactorizationTBD
            # so far I'm only dealing with ordinary and symmetric
            kind = issymmetric(aa_fact.matrixObj) ? SparseFactorizationCholesky :
                            SparseFactorizationQR
        end
        aa_fact._factorization = SparseFactor(kind, aa_fact.matrixObj.matrix)
        if aa_fact._factorization.status == SparseMatrixIsSingular
            throw(SingularException)
        elseif aa_fact._factorization.status == SparseStatusFailed
            throw(ErrorException("Factorization failed: check that the matrix"
                        * " has the correct properties for the factorization."))
        elseif aa_fact._factorization.status != SparseStatusOk
            throw(ErrorException("Something went wrong internally. Error type: "
                                * String(aa_fact._factorization.status)))
        end
    end
end

function solve(aa_fact::AAFactorization{T}, b::StridedVecOrMat{T}) where T<:vTypes
    @assert size(aa_fact.matrixObj)[2] == size(b, 1)
    factor!(aa_fact)
    x = Array{T}(undef, size(aa_fact.matrixObj)[2], size(b)[2:end]...)
    SparseSolve(aa_fact._factorization, b, x)
    return x
end

function solve!(aa_fact::AAFactorization{T}, xb::StridedVecOrMat{T}) where T<:vTypes
    @assert (xb isa StridedVector) ||
            (size(aa_fact.matrixObj)[1] == size(aa_fact.matrixObj)[2]) "Can't in-place " *
            "solve: x and b are different sizes and Julia cannot resize a matrix."
    factor!(aa_fact)
    SparseSolve(aa_fact._factorization, xb)
    return xb # because KLU.jl also returns
end

LinearAlgebra.ldiv!(aa_fact::AAFactorization{T}, xb::StridedVecOrMat{T}) where T<:vTypes =
        solve!(aa_fact, xb)

function LinearAlgebra.ldiv!(x::StridedVecOrMat{T},
                            aa_fact::AAFactorization{T},
                            b::StridedVecOrMat{T}) where T<:vTypes
    @assert size(aa_fact.matrixObj)[2] == size(b, 1)
    factor!(aa_fact)
    SparseSolve(aa_fact._factorization, b, x)
    return x
end

export AAFactorization, solve, solve!, AASparseMatrix, muladd!, factor!, factorize

end # module AASparseSolvers
