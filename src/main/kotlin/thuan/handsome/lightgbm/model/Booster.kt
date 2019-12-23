package thuan.handsome.lightgbm.model

import koma.matrix.Matrix

class Booster(dataset: Dataset, params: String) : CObject(), AutoCloseable {
    companion object {
        fun fit(params: Map<String, Any>, dataset: Dataset, rounds: Int): Booster {
            val objective = params.getOrDefault("objective", "<empty>")
            assert("binary" == objective) { "Objective $objective is not yet supported! Only 'binary' is supported for now, sorry!" }

            val booster = Booster(dataset, params.map { "${it.key}=${it.value}" }.joinToString(separator = " "))
            val intPointer = API.new_int32_tp()
            repeat(rounds) {
                API.LGBM_BoosterUpdateOneIter(booster.handle.getVoidSinglePointer(), intPointer)
            }
            API.delete_int32_tp(intPointer)
            return booster
        }
    }

    init {
        API.LGBM_BoosterCreate(dataset.handle.getVoidSinglePointer(), params, this.handle)
    }

    fun predict(row: DoubleArray): Double {
        val nativePointer = row.toNativeDoubleArray()
        val predPointer = API.new_doubleArray(1)
        val longPointer = 1.toLongPointer()

        API.LGBM_BoosterPredictForMatSingleRow(
            this.handle.getVoidSinglePointer(),
            nativePointer.toVoidPointer(),
            API.C_API_DTYPE_FLOAT64,
            row.size,
            1,
            API.C_API_PREDICT_NORMAL,
            0,
            "",
            longPointer,
            predPointer
        )

        val pred = API.doubleArray_getitem(predPointer, 0)
        API.delete_doubleArray(nativePointer)
        API.delete_doubleArray(predPointer)
        API.delete_int64_tp(longPointer)

        return pred
    }

    fun predict(data: Matrix<Double>): IntArray {
        val nativePointer = data.toNativeDoubleArray()

        val rows = data.numRows()
        val predPointer = API.new_doubleArray(rows)
        val longPointer = rows.toLongPointer()

        API.LGBM_BoosterPredictForMat(
            this.handle.getVoidSinglePointer(),
            nativePointer.toVoidPointer(),
            API.C_API_DTYPE_FLOAT64,
            rows,
            data.numCols(),
            1,
            API.C_API_PREDICT_NORMAL,
            0,
            "",
            longPointer,
            predPointer
        )

        val preds = IntArray(rows) {
            if (API.doubleArray_getitem(predPointer, it) >= 0.5) 1 else 0
        }

        API.delete_doubleArray(nativePointer)
        API.delete_doubleArray(predPointer)
        API.delete_int64_tp(longPointer)

        return preds
    }

    fun save(filePath: String) {
        API.LGBM_BoosterSaveModel(this.handle.getVoidSinglePointer(), 0, 0, filePath)
    }

    override fun close() {
        API.LGBM_BoosterFree(this.handle.getVoidSinglePointer())
    }
}
