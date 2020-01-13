package thuan.handsome.lightgbm

import koma.matrix.Matrix
import thuan.handsome.core.cv.crossValidate
import thuan.handsome.core.metrics.Metric
import thuan.handsome.core.predictor.Predictor
import thuan.handsome.core.utils.LOGGER

class Booster private constructor(dataset: Dataset, params: String) : CObject(), Predictor {
    companion object {
        fun fit(params: Map<String, Any>, data: Matrix<Double>, label: DoubleArray, rounds: Int): Booster {
            val dataset = Dataset.from(data, label)
            require(params.containsKey("objective")) {
                "Objective cannot be empty!!"
            }

            val objective = params["objective"]
            if (objective != "binary") {
                LOGGER.atWarning().log("Currently, only binary classification has been tested. Be careful!")
            }

            val booster = Booster(dataset, getParamsString(params))

            val intPointer = C_API.new_int32_tp()
            repeat(rounds) {
                C_API.LGBM_BoosterUpdateOneIter(booster.handle.getVoidSinglePointer(), intPointer)
            }

            // free system memory
            C_API.delete_int32_tp(intPointer)
            dataset.close()

            return booster
        }

        fun cv(
            metric: Metric,
            params: Map<String, Any>,
            data: Matrix<Double>,
            label: DoubleArray,
            maxiter: Int,
            nFolds: Int
        ): DoubleArray {
            return crossValidate(
                { X: Matrix<Double>, y: DoubleArray -> fit(params, X, y, maxiter) },
                metric, data, label, nFolds
            )
        }

        private fun getParamsString(params: Map<String, Any>): String {
            return params.map { "${it.key}=${it.value}" }.joinToString(separator = " ")
        }
    }

    init {
        C_API.LGBM_BoosterCreate(dataset.handle.getVoidSinglePointer(), params, this.handle)
    }

    override fun predict(data: Matrix<Double>): DoubleArray {
        val nativePointer = data.toNativeDoubleArray()

        val rows = data.numRows()
        val predPointer = C_API.new_doubleArray(rows)
        val longPointer = rows.toLongPointer()

        C_API.LGBM_BoosterPredictForMat(
            this.handle.getVoidSinglePointer(),
            nativePointer.toVoidPointer(),
            C_API.C_API_DTYPE_FLOAT64,
            rows,
            data.numCols(),
            1,
            C_API.C_API_PREDICT_NORMAL,
            0,
            "",
            longPointer,
            predPointer
        )

        val preds = DoubleArray(rows) {
            C_API.doubleArray_getitem(predPointer, it)
        }

        C_API.delete_doubleArray(nativePointer)
        C_API.delete_doubleArray(predPointer)
        C_API.delete_int64_tp(longPointer)

        return preds
    }

    override fun predict(x: DoubleArray): Double {
        val predPointer = C_API.new_doubleArray(1)
        val nativePointer = x.toNativeDoubleArray()
        val longPointer = 1.toLongPointer()

        C_API.LGBM_BoosterPredictForMatSingleRow(
            this.handle.getVoidSinglePointer(),
            nativePointer.toVoidPointer(),
            C_API.C_API_DTYPE_FLOAT64,
            x.size,
            1,
            C_API.C_API_PREDICT_NORMAL,
            0,
            "",
            longPointer,
            predPointer
        )

        val pred = C_API.doubleArray_getitem(predPointer, 0)
        C_API.delete_doubleArray(nativePointer)
        C_API.delete_doubleArray(predPointer)
        C_API.delete_int64_tp(longPointer)

        return pred
    }

    override fun save(filePath: String) {
        C_API.LGBM_BoosterSaveModel(this.handle.getVoidSinglePointer(), 0, 0, filePath)
    }

    override fun close() {
        C_API.LGBM_BoosterFree(this.handle.getVoidSinglePointer())
    }
}
