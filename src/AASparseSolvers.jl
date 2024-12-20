module AASparseSolvers

include("wrappers.jl")

# utility function mostly for testing.
# TODO: convert to a SparseArray instead.
function SparseToDense(A::SparseMatrix{T}) where T <: Union{Cdouble, Cfloat}
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

end # module AASparseSolvers
