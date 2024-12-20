# from: /Library/Developer/CommandLineTools/SDKs/MacOSX14.2.sdk/System/
# Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/
# vecLib.framework/Versions/A/Headers/Sparse/Solve.h

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
const att_type = Cuchar # I'm not sure if this is correct.
# it may not matter, though, for alignment reasons.
const ATT_TRANSPOSE = att_type(1)
const ATT_LOWER_TRIANGLE = att_type(0)
const ATT_UPPER_TRIANGLE = att_type(2)
const ATT_ORDINARY = att_type(0)
const ATT_TRIANGULAR = att_type(4)
const ATT_UNIT_TRIANGULAR = att_type(8)
const ATT_SYMMETRIC = att_type(12)


# mutable or no?
struct SparseMatrixStructure
    rowCount::Cint
    columnCount::Cint
    columnStarts::Ptr{Clong}
    rowIndices::Ptr{Cint}
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

struct SparseSymbolicFactorOptions
    control::SparseControl_t
    orderMethod::SparseOrder_t
    order::Ptr{Cvoid}
    ignoreRowsAndColumns::Ptr{Cvoid}
    # not sure how to write types of C-style function pointers in Julia.
    malloc::Ptr{Cvoid}#(::Csize_t)
    free::Ptr{Cvoid}#(::Ptr{Cvoid})
    reportError::Ptr{Cvoid}#(::Cstring) # assuming null-terminated.
end

mutable struct DenseVector{T<:Union{Cdouble, Cfloat}}
    count::Cint
    data::Ptr{T}
end

mutable struct SparseMatrix{T<:Union{Cdouble, Cfloat}}
    structure::SparseMatrixStructure
    data::Ptr{T}
end

mutable struct DenseMatrix{T<:Union{Cdouble, Cfloat}}
    rowCount::Cint
    columnCount::Cint
    columnStride::Cint
    attributes::att_type
    data::Ptr{T}
end

mutable struct SparseOpaqueSymbolicFactorization
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

mutable struct SparseOpaqueFactorization{T<:Union{Cdouble, Cfloat}}
    status::SparseStatus_t
    attributes::att_type
    symbolicFactorization::SparseOpaqueSymbolicFactorization
    userFactorStorage::Bool
    numericFactorization::Ptr{Cvoid}
    solveWorkspaceRequiredStatic::Csize_t
    solveWorkspaceRequiredPerRHS::Csize_t
end

# ignore for now: anything involving Subfactor, Preconditioner, or IterativeMethod

LIBSPARSE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/libSparse.dylib"


for T in (Cfloat, Cdouble)
    # simpler way that doesn't require @eval?
    # also, if I could write "dense vector-or-matrix," that'd cut the redundancy in half.
    local dmMultMangled = T == Cfloat ? :_Z14SparseMultiply18SparseMatrix_Float17DenseMatrix_FloatS0_ :
                        :_Z14SparseMultiply19SparseMatrix_Double18DenseMatrix_DoubleS0_
    @eval begin
        function SparseMultiply(arg1::SparseMatrix{$T}, arg2::DenseMatrix{$T}, arg3::DenseMatrix{$T})
            @ccall LIBSPARSE.$dmMultMangled(arg1::SparseMatrix{$T}, arg2::DenseMatrix{$T},
                                                arg3::DenseMatrix{$T})::Cvoid
        end
    end
    local dvMultMangled = T == Cfloat ? :_Z14SparseMultiply18SparseMatrix_Float17DenseVector_FloatS0_ :
                        :_Z14SparseMultiply19SparseMatrix_Double18DenseVector_DoubleS0_
    @eval begin
        function SparseMultiply(arg1::SparseMatrix{$T}, arg2::DenseVector{$T}, arg3::DenseVector{$T})
            @ccall LIBSPARSE.$dvMultMangled(arg1::SparseMatrix{$T}, arg2::DenseVector{$T},
                                            arg3::DenseVector{$T})::Cvoid
        end
    end

    local sdmMultMangled = T == Cfloat ? :_Z14SparseMultiplyf18SparseMatrix_Float17DenseMatrix_FloatS0_ :
                                :_Z14SparseMultiplyd19SparseMatrix_Double18DenseMatrix_DoubleS0_
    @eval begin
        function SparseMultiply(arg1::$T, arg2::SparseMatrix{$T}, arg3::DenseMatrix{$T}, arg4::DenseMatrix{$T})
            @ccall LIBSPARSE.$sdmMultMangled(arg1::$T, arg2::SparseMatrix{$T},
                                                arg3::DenseMatrix{$T}, arg4::DenseMatrix{$T})::Cvoid
        end
    end

    local sdvMultMangled = T == Cfloat ? :_Z14SparseMultiplyf18SparseMatrix_Float17DenseMatrix_FloatS0_ :
                                        :_Z14SparseMultiplyd19SparseMatrix_Double18DenseVector_DoubleS0_
    @eval begin
        function SparseMultiply(arg1::$T, arg2::SparseMatrix{$T}, arg3::DenseVector{$T}, arg4::DenseVector{$T})
            @ccall LIBSPARSE.$sdvMultMangled(arg1::$T, arg2::SparseMatrix{$T},
                                                arg3::DenseVector{$T}, arg4::DenseVector{$T})::Cvoid
        end
    end

    local dmMultAddMangled = T == Cfloat ? :_Z17SparseMultiplyAdd18SparseMatrix_Float17DenseMatrix_FloatS0_ :
                                            :_Z17SparseMultiplyAdd19SparseMatrix_Double18DenseMatrix_DoubleS0_
    @eval begin
        function SparseMultiplyAdd(arg1::SparseMatrix{$T}, arg2::DenseMatrix{$T}, arg3::DenseMatrix{$T})
            @ccall LIBSPARSE.$dmMultAddMangled(arg1::SparseMatrix{$T}, arg2::DenseMatrix{$T},
                                                arg3::DenseMatrix{$T})::Cvoid
        end
    end

    local dvMultAddMangled = T == Cfloat ? :_Z17SparseMultiplyAdd18SparseMatrix_Float17DenseVector_FloatS0_ :
                                            :_Z17SparseMultiplyAdd19SparseMatrix_Double18DenseVector_DoubleS0_
    @eval begin
        function SparseMultiplyAdd(arg1::SparseMatrix{$T}, arg2::DenseVector{$T}, arg3::DenseVector{$T})
            @ccall LIBSPARSE.$dvMultAddMangled(arg1::SparseMatrix{$T}, arg2::DenseVector{$T},
                                                arg3::DenseVector{$T})::Cvoid
        end
    end

    local sdmMultAddMangled = T == Cfloat ? :_Z17SparseMultiplyAddf18SparseMatrix_Float17DenseMatrix_FloatS0_ :
                                        :_Z17SparseMultiplyAddd19SparseMatrix_Double18DenseMatrix_DoubleS0_
    @eval begin
        function SparseMultiplyAdd(arg0::$T, arg1::SparseMatrix{$T}, arg2::DenseMatrix{$T}, arg3::DenseMatrix{$T})
            @ccall LIBSPARSE.$sdmMultAddMangled(arg0::$T, arg1::SparseMatrix{$T},
                                                arg2::DenseMatrix{$T}, arg3::DenseMatrix{$T})::Cvoid
        end
    end

    local sdvMultAddMangled = T == Cfloat ? :_Z17SparseMultiplyAddf18SparseMatrix_Float17DenseVector_FloatS0_ :
                                        :_Z17SparseMultiplyAddd19SparseMatrix_Double18DenseVector_DoubleS0_
    @eval begin
        function SparseMultiplyAdd(arg0::$T, arg1::SparseMatrix{$T}, arg2::DenseVector{$T}, arg3::DenseVector{$T})
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
        function SparseGetTranpose(arg1::SparseOpaqueFactorization{$T})
            @ccall LIBSPARSE.$ofTransposeMangled(arg1::SparseOpaqueFactorization{$T})::SparseOpaqueFactorization{$T}
        end
    end

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
end