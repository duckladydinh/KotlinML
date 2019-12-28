package thuan.handsome.gp.kernel

import koma.matrix.Matrix
import koma.ndarray.NDArray
import thuan.handsome.core.xspace.XSpace

interface Kernel {
    /**
     * @param dataX is a set of x (x:horizontal)
     * @param dataY is a set of y (y:horizontal)
     *
     * @return |dataX| by |dataY| covariance matrix between all pair of rows (x, y) and diagonal = 1...
     */
    fun getCovarianceMatrix(dataX: Matrix<Double>, dataY: Matrix<Double>, theta: DoubleArray): Matrix<Double>

    /**
     * @param data is a set of x (x:horizontal)
     *
     * @return |data| by |data| by |theta| covariance matrix between all pair of data's instances and its gradients
     */
    fun getCovarianceMatrixGradient(data: Matrix<Double>, theta: DoubleArray): NDArray<Double>

    /**
     * @param data is the matrix
     *
     * @return the diagonal of getCovarianceMatrix(data, theta)
     */
    fun getCovarianceMatrixTrace(data: Matrix<Double>, theta: DoubleArray): DoubleArray

    /**
     * @param data is a set of x (x:horizontal)
     *
     * @return |dataX| by |dataX| symmetric covariance matrix between all pair of rows (x, y) and diagonal = 1...
     */
    fun getCovarianceMatrix(data: Matrix<Double>, theta: DoubleArray): Matrix<Double>

    /**
     * @return the bound of the external control parameter. We do not use lengthScale directly
     * because exp is harder optimize. At least, that's my guess.
     */
    fun getThetaBounds(): XSpace

    /**
     * @return the number of parameters this kernel has
     */
    fun getDim(): Int
}
