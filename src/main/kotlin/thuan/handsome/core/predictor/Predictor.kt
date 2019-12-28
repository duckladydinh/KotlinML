package thuan.handsome.core.predictor

import koma.matrix.Matrix

interface Predictor {
    fun predict(data: Matrix<Double>): DoubleArray
    fun predict(x: DoubleArray): Double
    fun save(filePath: String)
}
