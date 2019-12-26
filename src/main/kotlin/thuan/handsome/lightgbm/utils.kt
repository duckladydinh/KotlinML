package thuan.handsome.lightgbm

import com.microsoft.ml.lightgbm.*
import koma.matrix.Matrix

typealias P_LONG = SWIGTYPE_p_long_long
typealias P_DOUBLE = SWIGTYPE_p_double
typealias PP_VOID = SWIGTYPE_p_p_void
typealias P_FLOAT = SWIGTYPE_p_float
typealias P_VOID = SWIGTYPE_p_void
typealias C_API = lightgbmlib

fun P_FLOAT.toVoidPointer(): P_VOID {
    return C_API.float_to_voidp_ptr(this)
}

fun P_DOUBLE.toVoidPointer(): P_VOID {
    return C_API.double_to_voidp_ptr(this)
}

fun PP_VOID.getVoidSinglePointer(): P_VOID {
    return C_API.voidpp_value(this)
}

fun <T : Number> T.toLongPointer(): P_LONG {
    val longPointer = C_API.new_int64_tp()
    C_API.int64_tp_assign(longPointer, this.toLong())
    return longPointer
}

fun FloatArray.toNativeFloatArray(): P_FLOAT {
    val nativePointer = C_API.new_floatArray(this.size)
    for ((index, value) in this.withIndex()) {
        C_API.floatArray_setitem(nativePointer, index, value)
    }
    return nativePointer
}

fun DoubleArray.toNativeDoubleArray(): P_DOUBLE {
    val nativePointer = C_API.new_doubleArray(this.size)
    for ((index, value) in this.withIndex()) {
        C_API.doubleArray_setitem(nativePointer, index, value)
    }
    return nativePointer
}

fun Matrix<Double>.toNativeDoubleArray(): P_DOUBLE {
    val nativePointer = C_API.new_doubleArray(this.size)

    this.toIterable().withIndex().forEach {
        C_API.doubleArray_setitem(nativePointer, it.index, it.value)
    }

    return nativePointer
}

fun IntArray.toFloatArray(): FloatArray {
    return FloatArray(this.size) {
        this[it].toFloat()
    }
}
