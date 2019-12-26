package thuan.handsome.lightgbm

import koma.matrix.Matrix

class Dataset private constructor() : CObject() {
    companion object {
        @JvmStatic
        fun from(data: Matrix<Double>, label: IntArray): Dataset {
            assert(label.isNotEmpty())
            assert(data.numRows() == label.size) {
                "The number of rows must equal the number of predictions. |X| = ${data.numRows()} != |y| = ${label.size}"
            }
            assert(label.min()!! >= 0) { "Only binary classification is currently supported" }
            assert(label.max()!! <= 1) { "Only binary classification is currently supported" }

            return Dataset().apply {
                setData(data)
                setLabel(label.toFloatArray())
            }
        }
    }

    private fun setLabel(label: FloatArray) {
        val nativePointer = label.toNativeFloatArray()
        C_API.LGBM_DatasetSetField(
            handle.getVoidSinglePointer(),
            "label",
            nativePointer.toVoidPointer(),
            label.size,
            C_API.C_API_DTYPE_FLOAT32
        )
        C_API.delete_floatArray(nativePointer)
    }

    private fun setData(data: Matrix<Double>) {
        val nativePointer = data.toNativeDoubleArray()
        C_API.LGBM_DatasetCreateFromMat(
            nativePointer.toVoidPointer(),
            C_API.C_API_DTYPE_FLOAT64,
            data.numRows(),
            data.numCols(),
            1,
            "",
            null,
            this.handle
        )
        C_API.delete_doubleArray(nativePointer)
    }

    override fun close() {
        C_API.LGBM_DatasetFree(this.handle.getVoidSinglePointer())
    }
}
