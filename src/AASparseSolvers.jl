module AASparseSolvers

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

mutable struct AASparseMatrix{T <: Union{Cfloat, Cdouble}}
    jl_data::SparseMatrixCSC{T, Int64}
    aa_matrix::SparseMatrix{T}
    # these don't *quite* match the jl_data's fields: 1 vs 0 indexing,
    # Cint vs Clong for row indices.
    _col_inds::Vector{Clong}
    _row_inds::Vector{Cint}
end

function AASparseMatrix(A::SparseMatrixCSC{T, Int64}) where T <: Union{Cfloat, Cdouble}
    res = AASparseMatrix(A, SparseMatrix{T}(), A.colptr .- 1,  Cint.(A.rowval) .- 1)
    res.aa_matrix.structure = SparseMatrixStructure(A.m, A.n, pointer(res._col_inds),
                                                pointer(res._row_inds), ATT_ORDINARY, 1)
    res.aa_matrix.data = pointer(A)
    return res
end

end # module AASparseSolvers
