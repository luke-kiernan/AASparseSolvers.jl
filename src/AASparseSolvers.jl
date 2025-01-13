module AASparseSolvers
using LinearAlgebra
include("wrappers.jl")

# utility function mostly for testing.
# TODO: convert to a SparseArray instead?
function SparseToMatrix(A::SparseMatrix{T}) where T <: Union{Cdouble, Cfloat}
    s = A.structure
    rows = s.blockSize*s.rowCount
    cols = s.blockSize*s.columnCount
    B = zeros(T, rows, cols)
    blockSize = s.blockSize
    for j in 1:s.columnCount
        # +1 for 0-indexed => 1-indexed.
        firstBlock = unsafe_load(s.columnStarts, j)+1
        lastBlock = unsafe_load(s.columnStarts, j+1)
        for k in firstBlock:lastBlock
            block_row = unsafe_load(s.rowIndices, k)
            # -1 for 1 indexed => 0-indexed.
            offset = blockSize*blockSize*(k-1)
            for (i,l) in Iterators.product(1:blockSize, 1:blockSize)
                offset += 1 # before we use the value: 0-indexed => 1-indexed.
                # j-1 for 1 indexed => 0 indexed.
                B[block_row*blockSize+i, (j-1)*blockSize+l] = unsafe_load(A.data, offset)
            end
        end
    end
    # If marked as symmetric, make symmetric.
    if s.attributes == ATT_SYMMETRIC | ATT_LOWER_TRIANGLE
        for c in 1:cols*blockSize
            for r in 1:c
                @assert(B[r,c] == 0.0 || B[r,c] == B[c,r])
                B[r,c] = B[c,r]
            end
        end
    elseif s.attributes == ATT_SYMMETRIC | ATT_UPPER_TRIANGLE
        for c in 1:cols*blockSize
            for r in c:(rows*blockSize)
                @assert(B[r,c] == 0.0 || B[r,c] == B[c,r])
                B[r,c] = B[c,r]
            end
        end
    end
    return B
end

function DenseToMatrix(A::DenseMatrix{T}) where T <: vTypes
    rows, cols = A.rowCount, A.columnCount
    B = zeros(T, rows, cols)
    r, c = 1, 1
    for i in 1:(rows*cols)
        B[r, c] = unsafe_load(A.data, i)
        if (r == rows)
            c += 1
            r = 1
        else
            r += 1
        end
    end
    return B
end

mutable struct AAFactorization{T<:vTypes}
    aa_matrix::SparseMatrix{T}
    # these don't *quite* match SparseArrayCSC fields: 1 vs 0 indexing,
    # Cint vs Clong for row indices.
    _col_inds::Vector{Clong}
    _row_inds::Vector{Cint}
    _data::Vector{T}
    _factorization::Union{Nothing, SparseOpaqueFactorization}
end

function AAFactorization(sparseM::SparseMatrixCSC{T, Int64})  where T<:vTypes
    c = Clong.(sparseM.colptr .+ -1)
    r = Cint.(sparseM.rowval .+ -1)
    vals = copy(sparseM.nzval)
    s = SparseMatrixStructure(size(sparseM)..., pointer(c), pointer(r), ATT_ORDINARY, 1)
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
        aa_fact._factorization = AASparseSolvers.SparseFactor(AASparseSolvers.SparseFactorizationQR, aa_fact.aa_matrix)
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
