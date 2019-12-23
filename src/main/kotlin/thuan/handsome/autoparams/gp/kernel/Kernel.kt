package thuan.handsome.autoparams.gp.kernel

import koma.matrix.Matrix

interface Kernel {
    fun evaluate(x: Matrix<Double>, y: Matrix<Double>): Double

    fun getCovarianceMatrix(X: Matrix<Double>): Matrix<Double>

    fun updateParameters(params: DoubleArray)

    fun getParameters(): DoubleArray

    fun getDimensions(): Int
}
