package thuan.handsome.lightgbm

import koma.matrix.Matrix
import thuan.handsome.core.utils.sliceByRows

class Booster private constructor(dataset: Dataset, params: String) : CObject() {
    companion object {
        fun fit(params: Map<String, Any>, data: Matrix<Double>, label: IntArray, rounds: Int): Booster {
            val dataset = Dataset.from(data, label)
            val objective = params.getOrDefault("objective", "<empty>")
            require("binary" == objective) {
                "Objective $objective is not yet supported! Only 'binary' is supported for now, sorry!"
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
            params: Map<String, Any>,
            data: Matrix<Double>,
            label: IntArray,
            maxiter: Int,
            nFolds: Int,
            metric: (IntArray, IntArray) -> Double
        ): DoubleArray {
            val folds = makeFolds(data.numRows(), nFolds)
            val scores = ArrayList<Double>(folds.size)

            for ((trainSet, validSet) in folds) {
                val trainData = sliceByRows(data, trainSet)
                val trainLabel = label.sliceArray(trainSet)
                val validData = sliceByRows(data, validSet)
                val validLabel = label.sliceArray(validSet)

                val booster = fit(params, trainData, trainLabel, maxiter)
                val preds = booster.predict(validData)
                booster.close()

                scores.add(metric.invoke(preds, validLabel))
            }

            return scores.toDoubleArray()
        }

        private fun getParamsString(params: Map<String, Any>) =
            params.map { "${it.key}=${it.value}" }.joinToString(separator = " ")

        private fun makeFolds(n: Int, nFolds: Int): List<Pair<List<Int>, List<Int>>> {
            val validSets = IntRange(0, n - 1).shuffled().withIndex()
                .groupBy { it.index % nFolds }
                .map { indexed ->
                    indexed.value.map {
                        it.value
                    }
                }

            val trainSets = ArrayList<List<Int>>(nFolds)
            for (i in validSets.indices) {
                val trainSet = validSets.slice(IntRange(0, validSets.size - 1) - i).flatten()
                trainSets.add(trainSet)
            }

            return trainSets zip validSets
        }
    }

    init {
        C_API.LGBM_BoosterCreate(dataset.handle.getVoidSinglePointer(), params, this.handle)
    }

    fun predict(row: DoubleArray): Double {
        val nativePointer = row.toNativeDoubleArray()
        val predPointer = C_API.new_doubleArray(1)
        val longPointer = 1.toLongPointer()

        C_API.LGBM_BoosterPredictForMatSingleRow(
            this.handle.getVoidSinglePointer(),
            nativePointer.toVoidPointer(),
            C_API.C_API_DTYPE_FLOAT64,
            row.size,
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

    fun predict(data: Matrix<Double>): IntArray {
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

        val preds = IntArray(rows) {
            if (C_API.doubleArray_getitem(predPointer, it) >= 0.5) 1 else 0
        }

        C_API.delete_doubleArray(nativePointer)
        C_API.delete_doubleArray(predPointer)
        C_API.delete_int64_tp(longPointer)

        return preds
    }

    fun save(filePath: String) {
        C_API.LGBM_BoosterSaveModel(this.handle.getVoidSinglePointer(), 0, 0, filePath)
    }

    override fun close() {
        C_API.LGBM_BoosterFree(this.handle.getVoidSinglePointer())
    }
}
