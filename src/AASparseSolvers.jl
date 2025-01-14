module AASparseSolvers
using LinearAlgebra
include("wrappers.jl")

mutable struct AAFactorization{T<:vTypes}
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
function AAFactorization(sparseM::SparseMatrixCSC{T, Int64},
                                symmetric::Bool = false, upperTriangle::Bool = false) where T<:vTypes
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

function Base.:(*)(A::AAFactorization{T}, x::Union{Matrix{T},Vector{T}}) where T<:vTypes
    @assert size(x)[1] == A.aa_matrix.structure.columnCount
    y = Array{T}(undef, A.aa_matrix.structure.rowCount, size(x)[2:end]...)
    if length(size(x)) == 2
        @assert size(y)[2] == size(x)[2]
    end
    SparseMultiply(A.aa_matrix, x, y)
    return y
end

function factor!(aa_fact::AAFactorization{T})  where T<:vTypes
    if isnothing(aa_fact._factorization)
        ordinary = aa_fact.aa_matrix.structure.attributes == AASparseSolvers.ATT_ORDINARY
        whichKind = ordinary ?  AASparseSolvers.SparseFactorizationQR :
                                    AASparseSolvers.SparseFactorizationLDLT
        aa_fact._factorization = AASparseSolvers.SparseFactor(whichKind, aa_fact.aa_matrix, true)
    end
end

function solve(aa_fact::AAFactorization{T}, b::Union{StridedMatrix{T}, StridedVector{T}}) where T<:vTypes
    @assert aa_fact.aa_matrix.structure.columnCount == size(b, 1)
    factor!(aa_fact)
    x = Array{T}(undef, aa_fact.aa_matrix.structure.columnCount, size(b)[2:end]...)
    AASparseSolvers.SparseSolve(aa_fact._factorization, b, x)
    return x
end

function solve!(aa_fact::AAFactorization{T}, xb::Union{StridedMatrix{T}, StridedVector{T}}) where T<:vTypes
    @assert (xb isa StridedVector) ||
            (aa_fact.aa_matrix.structure.rowCount == 
            aa_fact.aa_matrix.structure.columnCount) "Julia cannot resize a matrix"
    # Apple's library can handle non-square, but it's awkward with the Julia
    factor!(aa_fact)
    AASparseSolvers.SparseSolve(aa_fact._factorization, xb)
end

export AAFactorization, solve, solve!

end # module AASparseSolvers
