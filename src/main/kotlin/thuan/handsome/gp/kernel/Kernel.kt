package thuan.handsome.gp.kernel

import koma.matrix.Matrix
import koma.ndarray.NDArray

interface Kernel {
    var lowerBound: Double
    var upperBound: Double
    val dimension: Int

    fun getCovarianceMatrixGradient(data: Matrix<Double>, covMat: Matrix<Double>, theta: DoubleArray): NDArray<Double>
    fun getCovarianceMatrix(dataX: Matrix<Double>, dataY: Matrix<Double>, theta: DoubleArray): Matrix<Double>
    fun getCovarianceMatrix(data: Matrix<Double>, theta: DoubleArray): Matrix<Double>
}
