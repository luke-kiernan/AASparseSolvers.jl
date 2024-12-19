# from: /Library/Developer/CommandLineTools/SDKs/MacOSX14.2.sdk/System/
# Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/
# vecLib.framework/Versions/A/Headers/Sparse/Solve.h

module AASparseSolvers
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
    # simpler way that doesn't require quoting a symbol?
    # also, if I could write "dense vector-or-matrix," that'd cut the redundancy in half.
    local dmMultMangled = T == Cfloat ? :(:_Z14SparseMultiply18SparseMatrix_Float17DenseMatrix_FloatS0_) :
                        :(:_Z14SparseMultiply19SparseMatrix_Double18DenseMatrix_DoubleS0_)
    @eval begin
        function SparseMultiply(arg1::SparseMatrix{$T}, arg2::DenseMatrix{$T}, arg3::DenseMatrix{$T})
            ccall(($dmMultMangled, LIBSPARSE), Nothing,
                    (SparseMatrix{$T}, DenseMatrix{$T}, DenseMatrix{$T}),
                    arg1, arg2, arg3)
        end
    end
    local dvMultMangled = T == Cfloat ? :(:_Z14SparseMultiply18SparseMatrix_Float17DenseVector_FloatS0_) :
                        :(:_Z14SparseMultiply19SparseMatrix_Double18DenseVector_DoubleS0_)
    @eval begin
        function SparseMultiply(arg1::SparseMatrix{$T}, arg2::DenseVector{$T}, arg3::DenseVector{$T})
            ccall(($dvMultMangled, LIBSPARSE), Nothing,
                    (SparseMatrix{$T}, DenseVector{$T}, DenseVector{$T}),
                    arg1, arg2, arg3)
        end
    end

    local sdmMultMangled = T == Cfloat ? :(:_Z14SparseMultiplyf18SparseMatrix_Float17DenseMatrix_FloatS0_) :
                                :(:_Z14SparseMultiplyd19SparseMatrix_Double18DenseMatrix_DoubleS0_)
    @eval begin
        function SparseMultiply(arg1::$T, arg2::SparseMatrix{$T}, arg3::DenseMatrix{$T}, arg4::DenseMatrix{$T})
            ccall(($sdmMultMangled, LIBSPARSE), Nothing,
                    ($T, SparseMatrix{$T}, DenseMatrix{$T}, DenseMatrix{$T}),
                    arg1, arg2, arg3, arg4)
        end
    end

    local sdvMultMangled = T == Cfloat ? :(:_Z14SparseMultiplyf18SparseMatrix_Float17DenseMatrix_FloatS0_) :
                                        :(:_Z14SparseMultiplyd19SparseMatrix_Double18DenseVector_DoubleS0_)
    @eval begin
        function SparseMultiply(arg1::$T, arg2::SparseMatrix{$T}, arg3::DenseVector{$T}, arg4::DenseVector{$T})
            ccall(($sdvMultMangled, LIBSPARSE), Nothing,
                    ($T, SparseMatrix{$T}, DenseVector{$T}, DenseVector{$T}),
                    arg1, arg2, arg3, arg4)
        end
    end

    local dmMultAddMangled = T == Cfloat ? :(:_Z17SparseMultiplyAdd18SparseMatrix_Float17DenseMatrix_FloatS0_) :
                                            :(:_Z17SparseMultiplyAdd19SparseMatrix_Double18DenseMatrix_DoubleS0_)
    @eval begin
        function SparseMultiplyAdd(arg1::SparseMatrix{$T}, arg2::DenseMatrix{$T}, arg3::DenseMatrix{$T})
            ccall(($dmMultAddMangled, LIBSPARSE), Nothing,
                    (SparseMatrix{$T}, DenseMatrix{$T}, DenseMatrix{$T}),
                    arg1, arg2, arg3)
        end
    end

    local dvMultAddMangled = T == Cfloat ? :(:_Z17SparseMultiplyAdd18SparseMatrix_Float17DenseVector_FloatS0_) :
                                            :(:_Z17SparseMultiplyAdd19SparseMatrix_Double18DenseVector_DoubleS0_)
    @eval begin
        function SparseMultiplyAdd(arg1::SparseMatrix{$T}, arg2::DenseVector{$T}, arg3::DenseVector{$T})
            ccall(($dvMultAddMangled, LIBSPARSE), Nothing,
                    (SparseMatrix{$T}, DenseVector{$T}, DenseVector{$T}),
                    arg1, arg2, arg3)
        end
    end

    local sdmMultAddMangled = T == Cfloat ? :(:_Z17SparseMultiplyAddf18SparseMatrix_Float17DenseMatrix_FloatS0_) :
                                        :(:_Z17SparseMultiplyAddd19SparseMatrix_Double18DenseMatrix_DoubleS0_)
    @eval begin
        function SparseMultiplyAdd(arg0::$T, arg1::SparseMatrix{$T}, arg2::DenseMatrix{$T}, arg3::DenseMatrix{$T})
            ccall(($sdmMultAddMangled, LIBSPARSE), Nothing,
                    ($T, SparseMatrix{$T}, DenseMatrix{$T}, DenseMatrix{$T}),
                    arg0, arg1, arg2, arg3)
        end
    end

    local sdvMultAddMangled = T == Cfloat ? :(:_Z17SparseMultiplyAddf18SparseMatrix_Float17DenseVector_FloatS0_) :
                                        :(:_Z17SparseMultiplyAddd19SparseMatrix_Double18DenseVector_DoubleS0_)
    @eval begin
        function SparseMultiplyAdd(arg0::$T, arg1::SparseMatrix{$T}, arg2::DenseVector{$T}, arg3::DenseVector{$T})
            ccall(($sdvMultAddMangled, LIBSPARSE), Nothing,
                    ($T, SparseMatrix{$T}, DenseVector{$T}, DenseVector{$T}),
                    arg0, arg1, arg2, arg3)
        end
    end

    local mTransposeMangled = T == Cfloat ? :(:_Z18SparseGetTranspose18SparseMatrix_Float) :
                                            :(:_Z18SparseGetTranspose19SparseMatrix_Double)
    @eval begin
        function SparseGetTranspose(arg1::SparseMatrix{$T})
            ccall(($mTransposeMangled, LIBSPARSE), SparseMatrix{$T},
                    (SparseMatrix{$T},), arg1)
        end
    end
    # skipped: 2 subfactor transposes.
    local ofTransposeMangled = T == Cfloat ? :(:_Z18SparseGetTranspose31SparseOpaqueFactorization_Float) :
                                        :(:_Z18SparseGetTranspose32SparseOpaqueFactorization_Double)
    @eval begin
        function SparseGetTranpose(arg1::SparseOpaqueFactorization{$T})
            ccall(($ofTransposeMangled, LIBSPARSE), SparseOpaqueFactorization{$T},
                    (SparseOpaqueFactorization{$T},), arg1)
        end
    end

    local convertCoordMangled = T == Cfloat ? :(:_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKf) :
                                            :(:_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKd)
    @eval begin
        function SparseConvertFromCoord(arg1::Cint, arg2::Cint, arg3::Clong, arg4::Cuchar, arg5::$att_type,
                                        arg6::Ptr{Cint}, arg7::Ptr{Cint}, arg8::Ptr{$T})
            ccall(($convertCoordMangled, LIBSPARSE), SparseMatrix{$T},
                    (Cint, Cint, Clong, Cuchar, $att_type, Ptr{Cint}, Ptr{Cint}, Ptr{$T}),
                    arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
        end
    end

    local uConvertCoordMangled = T == Cfloat ? :(:_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKdPvS4_) :
                                            :(:_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKfPvS4_)
    @eval begin
        function SparseConvertFromCoord(arg1::Cint, arg2::Cint, arg3::Clong, arg4::Cuchar, arg5::$att_type,
                    arg6::Ptr{Cint}, arg7::Ptr{Cint}, arg8::Ptr{$T}, arg9::Ptr{Cvoid}, arg10::Ptr{Cvoid})
            ccall(($uConvertCoordMangled, LIBSPARSE), SparseMatrix{$T},
                    (Cint, Cint, Clong, Cuchar, $att_type, Ptr{Cint}, Ptr{Cint}, Ptr{$T}, Ptr{Cvoid}, Ptr{Cvoid}),
                    arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
        end
    end
end

end