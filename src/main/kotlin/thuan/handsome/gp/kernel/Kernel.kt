package thuan.handsome.gp.kernel

import koma.matrix.Matrix
import koma.ndarray.NDArray
import thuan.handsome.core.xspace.XSpace

interface Kernel {
    fun getCovarianceMatrixGradient(data: Matrix<Double>, covMat: Matrix<Double>, theta: DoubleArray): NDArray<Double>
    fun getCovarianceMatrixTrace(dataX: Matrix<Double>, dataY: Matrix<Double>, theta: DoubleArray): DoubleArray
    fun getCovarianceMatrix(dataX: Matrix<Double>, dataY: Matrix<Double>, theta: DoubleArray): Matrix<Double>
    fun getCovarianceMatrix(data: Matrix<Double>, theta: DoubleArray): Matrix<Double>
    fun getThetaBounds(): XSpace
}
