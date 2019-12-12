package thuan.handsome.lightgbm.model

import com.microsoft.ml.lightgbm.*
import thuan.handsome.lightgbm.env.NativeLoader

typealias VoidDoublePointer = SWIGTYPE_p_p_void
typealias LongPointer = SWIGTYPE_p_long_long
typealias DoublePointer = SWIGTYPE_p_double
typealias FloatPointer = SWIGTYPE_p_float
typealias VoidPointer = SWIGTYPE_p_void
typealias API = lightgbmlib

fun initNative() {
	val osPrefix = NativeLoader.osPrefix
	NativeLoader("/com/microsoft/ml/lightgbm").loadLibraryByName(osPrefix + "_lightgbm")
	NativeLoader("/com/microsoft/ml/lightgbm").loadLibraryByName(osPrefix + "_lightgbm_swig")
}

fun FloatPointer.toVoidPointer(): VoidPointer {
	return API.float_to_voidp_ptr(this)
}

fun DoublePointer.toVoidPointer(): VoidPointer {
	return API.double_to_voidp_ptr(this)
}

fun VoidDoublePointer.getVoidSinglePointer(): VoidPointer {
	return API.voidpp_value(this)
}

fun <T : Number> T.toLongPointer(): LongPointer {
	val longPointer = API.new_int64_tp()
	API.int64_tp_assign(longPointer, this.toLong())
	return longPointer
}

fun FloatArray.toNativeFloatArray(): FloatPointer {
	val nativePointer = API.new_floatArray(this.size)
	for ((index, value) in this.withIndex()) {
		API.floatArray_setitem(nativePointer, index, value)
	}
	return nativePointer
}

fun DoubleArray.toNativeDoubleArray(): DoublePointer {
	val nativePointer = API.new_doubleArray(this.size)
	for ((index, value) in this.withIndex()) {
		API.doubleArray_setitem(nativePointer, index, value)
	}
	return nativePointer
}

fun Array<DoubleArray>.toNativeDoubleArray(): DoublePointer {
	val rows = this.size
	val cols = this[0].size
	val nativePointer = API.new_doubleArray(rows * cols)

	for ((rowIndex, row) in this.withIndex()) {
		for ((index, value) in row.withIndex()) {
			API.doubleArray_setitem(nativePointer, rowIndex * cols + index, value)
		}
	}
	return nativePointer
}

fun IntArray.toFloatArray(): FloatArray {
	return FloatArray(this.size) {
		this[it].toFloat()
	}
}