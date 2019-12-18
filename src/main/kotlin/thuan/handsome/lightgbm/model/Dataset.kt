package thuan.handsome.lightgbm.model

import koma.matrix.Matrix
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Dataset : CObject() {
	companion object {
		@JvmStatic
		fun from(data: Matrix<Double>, label: IntArray): Dataset {
			assertTrue(label.isNotEmpty())
			assertEquals(
				data.numRows(), label.size,
				"The number of rows must equal the number of predictions. |X| = ${data.numRows()} != |y| = ${label.size}"
			)
			assertTrue(label.min()!! >= 0, "Only binary classification is currently supported")
			assertTrue(label.max()!! <= 1, "Only binary classification is currently supported")

			return Dataset().apply {
				setData(data)
				setLabel(label.toFloatArray())
			}
		}
	}

	private fun setLabel(label: FloatArray) {
		val nativePointer = label.toNativeFloatArray()
		API.LGBM_DatasetSetField(
			handle.getVoidSinglePointer(),
			"label",
			nativePointer.toVoidPointer(),
			label.size,
			API.C_API_DTYPE_FLOAT32
		)
		API.delete_floatArray(nativePointer)
	}

	private fun setData(data: Matrix<Double>) {
		val nativePointer = data.toNativeDoubleArray()
		API.LGBM_DatasetCreateFromMat(
			nativePointer.toVoidPointer(),
			API.C_API_DTYPE_FLOAT64,
			data.numRows(),
			data.numCols(),
			1,
			"",
			null,
			this.handle
		)
		API.delete_doubleArray(nativePointer)
	}

	override fun close() {
		API.LGBM_DatasetFree(this.handle.getVoidSinglePointer())
	}
}