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

# this really ought to operate on a matrix wrapper, not a factorization...
# write a more polished wrapper and replace aa_matrix with that?
function Base.:(*)(A::AAFactorization{T}, x::Union{Matrix{T},Vector{T}}) where T<:vTypes
    @assert size(x)[1] == A.aa_matrix.structure.columnCount
    y = Array{T}(undef, A.aa_matrix.structure.rowCount, size(x)[2:end]...)
    if length(size(x)) == 2
        @assert size(y)[2] == size(x)[2]
    end
    SparseMultiply(A.aa_matrix, x, y)
    return y
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
            aa_fact.aa_matrix.structure.columnCount) "Can't in-place solve:" *
            " x and b are different sizes and Julia cannot resize a matrix."
    factor!(aa_fact)
    AASparseSolvers.SparseSolve(aa_fact._factorization, xb)
    return xb # because KLU also returns
end

# TODO: ldiv!

export AAFactorization, solve, solve!

end # module AASparseSolvers
