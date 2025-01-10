using SparseArrays

# useful for debugging field offsets.
#= name(::Type{T}) where {T} = (isempty(T.parameters) ? T : T.name.wrapper)
function PrintAlignments(T::Type)
    println(name(T), " is size ", sizeof(T))
    for i in 1:fieldcount(T)
        println(fieldoffset(T, i), " ",
                fieldname(T, i), " ",
                sizeof(fieldtype(T, i)))
    end
end =#

@enum SparseTriangle_t::UInt8 begin
    SparseLowerTriangle = 0
    SparseUpperTriangle = 1
end

@enum SparseKind_t::UInt32 begin
    SparseOrdinary = 0
    SparseTriangular = 1
    SparseUnitTriangular = 2
    SparseSymmetric = 4
end

@enum SparseFactorization_t::UInt8 begin
    SparseFactorizationCholesky = 0
    SparseFactorizationLDLT = 1
    SparseFactorizationLDLTUnpivoted = 2
    SparseFactorizationLDLTSBK = 3
    SparseFactorizationLDLTTPP = 4
    SparseFactorizationQR = 40
    SparseFactorizationCholeskyAtA = 41
end

@enum SparseOrder_t::UInt8 begin
    SparseOrderDefault = 0
    SparseOrderUser = 1
    SparseOrderAMD = 2
    SparseOrderMetis = 3
    SparseOrderCOLAMD = 4
end

@enum SparseScaling_t::UInt8 begin
    SpraseScalingDefault = 0
    SparseScalingUser = 1
    SparseScalingQuilibriationInf = 2
end

@enum SparseStatus_t::Int32 begin
    SparseStatusOk = 0
    SparseStatusFailed = -1
    SparseMatrixIsSingular = -2
    SparseInternalError = -3
    SparseParameterError = -4
    SparseStatusReleased = -2147483647
end

@enum SparseControl_t::UInt32 begin
    SparseDefaultControl = 0
end

# can't pack bitflags in raw Julia, so can't implement SparseAttributes
# directly. Workaround with flags:
const att_type = Cuint
const ATT_TRANSPOSE = att_type(1)
const ATT_LOWER_TRIANGLE = att_type(0)
const ATT_UPPER_TRIANGLE = att_type(2)
const ATT_ORDINARY = att_type(0)
const ATT_TRIANGULAR = att_type(4)
const ATT_UNIT_TRIANGULAR = att_type(8)
const ATT_SYMMETRIC = att_type(12)
const ATT_ALLOCATED_BY_SPARSE = att_type(1) << 15

const vTypes = Union{Cfloat, Cdouble}

struct SparseMatrixStructure
    rowCount::Cint
    columnCount::Cint
    columnStarts::Ptr{Clong} # Apple uses different types for indices
    rowIndices::Ptr{Cint} # whereas SparseMatrixCSC uses the same
    attributes::att_type
    blockSize::UInt8
end

struct SparseNumericFactorOptions
    control::SparseControl_t
    scalingMethod::SparseScaling_t
    scaling::Ptr{Cvoid}
    pivotTolerance::Float64
    zeroTolerance::Float64
end

# Note: to use system defaults, values of malloc/free should be
# nil, which is *not the same* as C_NULL.
struct SparseSymbolicFactorOptions
    control::SparseControl_t
    orderMethod::SparseOrder_t
    order::Ptr{Cvoid}
    ignoreRowsAndColumns::Ptr{Cvoid}
    malloc::Ptr{Cvoid} # arg: Csize_t
    free::Ptr{Cvoid} # arg: Ptr{Cvoid}
    reportError::Ptr{Cvoid} # arg: Cstring, assuming null-terminated.
end

# attempt at error handling (see above about nil vs C_NULL)
#=
CError(text::Cstring)::Cvoid = error(unsafe_string(text))
CErrorPtr = @cfunction(CError, Cvoid, (Cstring, ))
=#

struct DenseVector{T<:vTypes}
    count::Cint
    data::Ptr{T}
end

struct SparseMatrix{T<:vTypes}
    structure::SparseMatrixStructure
    data::Ptr{T}
end

struct DenseMatrix{T<:vTypes}
    rowCount::Cint
    columnCount::Cint
    columnStride::Cint
    attributes::att_type
    data::Ptr{T}
end

# must be immutable.
struct SparseOpaqueSymbolicFactorization
    status::SparseStatus_t 
    rowCount::Cint 
    columnCount::Cint
    attributes::att_type
    blockSize::UInt8
    type::SparseFactorization_t
    factorization::Ptr{Cvoid}
    workspaceSize_Float::Csize_t
    workspaceSize_Double::Csize_t
    factorSize_Float::Csize_t
    factorSize_Double::Csize_t
end
# const SymbolicFactor = SparseOpaqueSymbolicFactorization

# TODO: I have T here to match the C: _Double, _Float variants. But T isn't used!!
struct SparseOpaqueFactorization{T<:vTypes}
    status::SparseStatus_t
    attributes::att_type
    symbolicFactorization::SparseOpaqueSymbolicFactorization
    userFactorStorage::Bool
    numericFactorization::Ptr{Cvoid}
    solveWorkspaceRequiredStatic::Csize_t
    solveWorkspaceRequiredPerRHS::Csize_t
end

# const Factor{T} = SparseOpaqueFactorization{T}
# ignore for now: anything involving Subfactor, Preconditioner, or IterativeMethod

const LIBSPARSE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/libSparse.dylib"

Base.cconvert(::Type{DenseMatrix{T}}, m::Matrix{T}) where T<:vTypes = m

function Base.unsafe_convert(::Type{DenseMatrix{T}}, m::Matrix{T}) where T<:vTypes
    return DenseMatrix{T}(size(m)..., size(m, 1), ATT_ORDINARY, pointer(m))
end

Base.cconvert(::Type{DenseVector{T}}, v::Vector{T}) where T<:vTypes = v

function Base.unsafe_convert(::Type{DenseVector{T}}, v::Vector{T}) where T<:vTypes
    return DenseVector{T}(size(v)[1], pointer(v))
end

for T in (Cfloat, Cdouble)
    # TODO: simpler way that doesn't require @eval?
    # also, if I could write "dense vector-or-matrix," that'd cut the redundancy in half.
    # TODO: would be nice to find a way to avoid copy-pasting Union{DenseMatrix{$T}, Matrix{$T}}. 
    local dmMultMangled = T == Cfloat ? :_Z14SparseMultiply18SparseMatrix_Float17DenseMatrix_FloatS0_ :
                        :_Z14SparseMultiply19SparseMatrix_Double18DenseMatrix_DoubleS0_
    @eval begin
        function SparseMultiply(arg1::SparseMatrix{$T}, arg2::Union{DenseMatrix{$T}, Matrix{$T}},
                                                            arg3::Union{DenseMatrix{$T}, Matrix{$T}})
            @ccall LIBSPARSE.$dmMultMangled(arg1::SparseMatrix{$T}, arg2::DenseMatrix{$T},
                                                arg3::DenseMatrix{$T})::Cvoid
        end
    end

    local dvMultMangled = T == Cfloat ? :_Z14SparseMultiply18SparseMatrix_Float17DenseVector_FloatS0_ :
                        :_Z14SparseMultiply19SparseMatrix_Double18DenseVector_DoubleS0_
    @eval begin
        function SparseMultiply(arg1::SparseMatrix{$T}, arg2::Union{DenseVector{$T}, Vector{$T}},
                                                        arg3::Union{DenseVector{$T}, Vector{$T}})
            @ccall LIBSPARSE.$dvMultMangled(arg1::SparseMatrix{$T}, arg2::DenseVector{$T},
                                            arg3::DenseVector{$T})::Cvoid
        end
    end

    local sdmMultMangled = T == Cfloat ? :_Z14SparseMultiplyf18SparseMatrix_Float17DenseMatrix_FloatS0_ :
                                :_Z14SparseMultiplyd19SparseMatrix_Double18DenseMatrix_DoubleS0_
    @eval begin
        function SparseMultiply(arg1::$T, arg2::SparseMatrix{$T}, arg3::Union{DenseMatrix{$T}, Matrix{$T}},
                                                    arg4::Union{DenseMatrix{$T}, Matrix{$T}})
            @ccall LIBSPARSE.$sdmMultMangled(arg1::$T, arg2::SparseMatrix{$T},
                                                arg3::DenseMatrix{$T}, arg4::DenseMatrix{$T})::Cvoid
        end
    end

    local sdvMultMangled = T == Cfloat ? :_Z14SparseMultiplyf18SparseMatrix_Float17DenseVector_FloatS0_ :
                                        :_Z14SparseMultiplyd19SparseMatrix_Double18DenseVector_DoubleS0_
    @eval begin
        function SparseMultiply(arg1::$T, arg2::SparseMatrix{$T}, arg3::Union{DenseVector{$T}, Vector{$T}},
                                                    arg4::Union{DenseVector{$T}, Vector{$T}})
            @ccall LIBSPARSE.$sdvMultMangled(arg1::$T, arg2::SparseMatrix{$T},
                                                arg3::DenseVector{$T}, arg4::DenseVector{$T})::Cvoid
        end
    end

    local dmMultAddMangled = T == Cfloat ? :_Z17SparseMultiplyAdd18SparseMatrix_Float17DenseMatrix_FloatS0_ :
                                            :_Z17SparseMultiplyAdd19SparseMatrix_Double18DenseMatrix_DoubleS0_
    @eval begin
        function SparseMultiplyAdd(arg1::SparseMatrix{$T}, arg2::Union{DenseMatrix{$T}, Matrix{$T}},
                                                arg3::Union{DenseMatrix{$T}, Matrix{$T}})
            @ccall LIBSPARSE.$dmMultAddMangled(arg1::SparseMatrix{$T}, arg2::DenseMatrix{$T},
                                                arg3::DenseMatrix{$T})::Cvoid
        end
    end

    local dvMultAddMangled = T == Cfloat ? :_Z17SparseMultiplyAdd18SparseMatrix_Float17DenseVector_FloatS0_ :
                                            :_Z17SparseMultiplyAdd19SparseMatrix_Double18DenseVector_DoubleS0_
    @eval begin
        function SparseMultiplyAdd(arg1::SparseMatrix{$T}, arg2::Union{DenseVector{$T}, Vector{$T}},
                                                                arg3::Union{DenseVector{$T}, Vector{$T}})
            @ccall LIBSPARSE.$dvMultAddMangled(arg1::SparseMatrix{$T}, arg2::DenseVector{$T},
                                                arg3::DenseVector{$T})::Cvoid
        end
    end

    local sdmMultAddMangled = T == Cfloat ? :_Z17SparseMultiplyAddf18SparseMatrix_Float17DenseMatrix_FloatS0_ :
                                        :_Z17SparseMultiplyAddd19SparseMatrix_Double18DenseMatrix_DoubleS0_
    @eval begin
        function SparseMultiplyAdd(arg0::$T, arg1::SparseMatrix{$T}, arg2::Union{DenseMatrix{$T}, Matrix{$T}},
                                                                arg3::Union{DenseMatrix{$T}, Matrix{$T}})
            @ccall LIBSPARSE.$sdmMultAddMangled(arg0::$T, arg1::SparseMatrix{$T},
                                                arg2::DenseMatrix{$T}, arg3::DenseMatrix{$T})::Cvoid
        end
    end

    local sdvMultAddMangled = T == Cfloat ? :_Z17SparseMultiplyAddf18SparseMatrix_Float17DenseVector_FloatS0_ :
                                        :_Z17SparseMultiplyAddd19SparseMatrix_Double18DenseVector_DoubleS0_
    @eval begin
        function SparseMultiplyAdd(arg0::$T, arg1::SparseMatrix{$T}, arg2::Union{DenseVector{$T}, Vector{$T}},
                                                                        arg3::Union{DenseVector{$T}, Vector{$T}})
            @ccall LIBSPARSE.$sdvMultAddMangled(arg0::$T, arg1::SparseMatrix{$T},
                                                arg2::DenseVector{$T}, arg3::DenseVector{$T})::Cvoid
        end
    end

    local mTransposeMangled = T == Cfloat ? :_Z18SparseGetTranspose18SparseMatrix_Float :
                                            :_Z18SparseGetTranspose19SparseMatrix_Double
    @eval begin
        function SparseGetTranspose(arg1::SparseMatrix{$T})
            @ccall LIBSPARSE.$mTransposeMangled(arg1::SparseMatrix{$T})::SparseMatrix{$T}
        end
    end
    # skipped: 2 subfactor transposes.
    local ofTransposeMangled = T == Cfloat ? :_Z18SparseGetTranspose31SparseOpaqueFactorization_Float :
                                        :_Z18SparseGetTranspose32SparseOpaqueFactorization_Double
    @eval begin
        function SparseGetTranspose(arg1::SparseOpaqueFactorization{$T})
            @ccall LIBSPARSE.$ofTransposeMangled(arg1::SparseOpaqueFactorization{$T})::SparseOpaqueFactorization{$T}
        end
    end
    # TODO: these SparseConvertFromCoord functions are untested.
    local convertCoordMangled = T == Cfloat ? :_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKf :
                                            :_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKd
    @eval begin
        function SparseConvertFromCoord(arg1::Cint, arg2::Cint, arg3::Clong, arg4::Cuchar, arg5::$att_type,
                                        arg6::Ptr{Cint}, arg7::Ptr{Cint}, arg8::Ptr{$T})
            @ccall LIBSPARSE.$convertCoordMangled(arg1::Cint, arg2::Cint, arg3::Clong, arg4::Cuchar, arg5::$att_type,
                                                    arg6::Ptr{Cint}, arg7::Ptr{Cint}, arg8::Ptr{$T})::SparseMatrix{$T}
        end
    end

    local uConvertCoordMangled = T == Cfloat ? :_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKdPvS4_ :
                                            :_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKfPvS4_
    @eval begin
        function SparseConvertFromCoord(arg1::Cint, arg2::Cint, arg3::Clong, arg4::Cuchar, arg5::$att_type,
                    arg6::Ptr{Cint}, arg7::Ptr{Cint}, arg8::Ptr{$T}, arg9::Ptr{Cvoid}, arg10::Ptr{Cvoid})
            @ccall LIBSPARSE.$uConvertCoordMangled(arg1::Cint, arg2::Cint, arg3::Clong, arg4::Cuchar, arg5::$att_type,
                    arg6::Ptr{Cint}, arg7::Ptr{Cint}, arg8::Ptr{$T}, arg9::Ptr{Cvoid}, arg10::Ptr{Cvoid})::SparseMatrix{$T}
        end
    end

    local mCleanup = T == Cfloat ? :_Z13SparseCleanup18SparseMatrix_Float :
                                     :_Z13SparseCleanup19SparseMatrix_Double
    @eval SparseCleanup(arg1::SparseMatrix{$T}) = @ccall (
            LIBSPARSE.$mCleanup(arg1::SparseMatrix{$T})::Cvoid
    )
    
    local ofCleanup = T == Cfloat ? :_Z13SparseCleanup31SparseOpaqueFactorization_Float :
                                    :_Z13SparseCleanup32SparseOpaqueFactorization_Double
    @eval SparseCleanup(arg1::SparseOpaqueFactorization{$T}) = @ccall (
            LIBSPARSE.$ofCleanup(arg1::SparseOpaqueFactorization{$T})::Cvoid
    )

    local sparseFactorMatrix = T == Cfloat ? :_Z12SparseFactorh18SparseMatrix_Float :
                                                :_Z12SparseFactorh19SparseMatrix_Double
    @eval begin
        function SparseFactor(arg1::SparseFactorization_t, arg2::SparseMatrix{$T})::SparseOpaqueFactorization{$T}
            @ccall LIBSPARSE.$sparseFactorMatrix(arg1::SparseFactorization_t, arg2::SparseMatrix{$T})::SparseOpaqueFactorization{$T}
        end
    end

    local sparseSolveInplace = T == Cfloat ? :_Z11SparseSolve31SparseOpaqueFactorization_Float17DenseMatrix_Float :
                                            :_Z11SparseSolve32SparseOpaqueFactorization_Double18DenseMatrix_Double
    @eval SparseSolve(arg1::SparseOpaqueFactorization{$T}, arg2::Union{DenseMatrix{$T}, Matrix{$T}}) = @ccall (
        LIBSPARSE.$sparseSolveInplace(arg1::SparseOpaqueFactorization{$T}, arg2::DenseMatrix{$T})::Cvoid
    )

    local sparseSolve = T == Cfloat ? :_Z11SparseSolve31SparseOpaqueFactorization_Float17DenseMatrix_FloatS0_ :
                                :_Z11SparseSolve32SparseOpaqueFactorization_Double18DenseMatrix_DoubleS0_
    @eval SparseSolve(arg1::SparseOpaqueFactorization{$T}, arg2::Union{DenseMatrix{$T}, Matrix{$T}},
                        arg3::Union{DenseMatrix{$T}, Matrix{$T}}) = @ccall (
        LIBSPARSE.$sparseSolve(arg1::SparseOpaqueFactorization{$T}, arg2::DenseMatrix{$T}, 
                                    arg3::DenseMatrix{$T})::Cvoid
    )

    local sparseSolveVecInPlace = T == Cfloat ? :_Z11SparseSolve31SparseOpaqueFactorization_Float17DenseVector_Float :
                                :_Z11SparseSolve32SparseOpaqueFactorization_Double18DenseVector_Double
    @eval SparseSolve(arg1::SparseOpaqueFactorization{$T},
                    arg2::Union{DenseVector{$T}, Vector{$T}}) = @ccall (
        LIBSPARSE.$sparseSolveVecInPlace(arg1::SparseOpaqueFactorization{$T},
                                            arg2::DenseVector{$T})::Cvoid
    )

    local sparseSolveVec = T == Cfloat ? :_Z11SparseSolve31SparseOpaqueFactorization_Float17DenseVector_FloatS0_ :
                                    :_Z11SparseSolve32SparseOpaqueFactorization_Double18DenseVector_DoubleS0_
    @eval SparseSolve(arg1::SparseOpaqueFactorization{$T},
                    arg2::Union{DenseVector{$T}, Vector{$T}},
                    arg3::Union{DenseVector{$T}, Vector{$T}}) = @ccall (
        LIBSPARSE.$sparseSolveVec(arg1::SparseOpaqueFactorization{$T},
                                arg2::DenseVector{$T}, arg3::DenseVector{$T})::Cvoid
    )
end
SparseCleanup(arg1::SparseOpaqueSymbolicFactorization) = @ccall (
        LIBSPARSE._Z13SparseCleanup33SparseOpaqueSymbolicFactorization(
                                arg1::SparseOpaqueSymbolicFactorization)::Cvoid
)
SparseFactor(arg1::SparseFactorization_t, arg2::SparseMatrixStructure) = @ccall (
    LIBSPARSE._Z12SparseFactorh21SparseMatrixStructure(
        arg1::SparseFactorization_t, arg2::SparseMatrixStructure
    )::SparseOpaqueSymbolicFactorization
)

# attempt at error handling (see above about nil vs C_NULL)
#= SparseFactor(arg1::SparseFactorization_t,
            arg2::SparseMatrixStructure) = SparseFactor(
                arg1, arg2, CErrorPtr)

SparseFactor(arg1::SparseFactorization_t, arg2::SparseMatrixStructure,
            arg3::SparseSymbolicFactorOptions) = @ccall(
     LIBSPARSE._Z12SparseFactorh21SparseMatrixStructure27SparseSymbolicFactorOptions(
        arg1::Cuint, arg2::SparseMatrixStructure,
            arg3::SparseSymbolicFactorOptions
    )::SparseOpaqueSymbolicFactorization
) =#