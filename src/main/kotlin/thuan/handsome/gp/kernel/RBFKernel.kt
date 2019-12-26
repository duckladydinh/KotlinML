package thuan.handsome.gp.kernel

import koma.extensions.get
import koma.extensions.set
import koma.matrix.Matrix
import koma.ndarray.NDArray
import koma.pow
import koma.zeros
import kotlin.math.exp
import kotlin.math.ln
import thuan.handsome.core.xspace.*

class RBFKernel private constructor(private val bounds: Array<Bound>) : Kernel {
    constructor(dimension: Int) : this(Bound(1e-5, 1e5).times(dimension))

    override fun getCovarianceMatrixTrace(
        dataX: Matrix<Double>,
        dataY: Matrix<Double>,
        theta: DoubleArray
    ): DoubleArray {
        return DoubleArray(dataX.numRows()) { 1.0 }
    }

    /**
	 * @param dataX is a set of x (x:horizontal)
	 * @param dataY is a set of y (y:horizontal)
	 *
	 * @return |dataX| by |dataY| covariance symmetric matrix between all pair of (x, y) and diagonal = 1...
	 */
    override fun getCovarianceMatrix(dataX: Matrix<Double>, dataY: Matrix<Double>, theta: DoubleArray): Matrix<Double> {
        val n = dataX.numRows()
        val m = dataY.numRows()
        val covMat = zeros(n, m)
        val lengthScale = getLengthScale(theta)

        for (i in 0 until n) {
            for (j in i + 1 until m) {
                covMat[i, j] = distance(dataX.getRow(i).transpose(), dataY.getRow(j).transpose(), lengthScale)
                covMat[j, i] = covMat[i, j]
            }
            if (n == m) {
                covMat[i, i] = 1
            }
        }
        return covMat
    }

    /**
	 * @param data is a set of x (x:horizontal)
	 *
	 * @return |data| by |data| covariance matrix between all pair of data's instances
	 */
    override fun getCovarianceMatrix(data: Matrix<Double>, theta: DoubleArray): Matrix<Double> {
        return getCovarianceMatrix(data, data, theta)
    }

    override fun getThetaBounds(): XSpace {
        val xSpace = UniformXSpace()
        for ((i, bound) in bounds.withIndex()) {
            xSpace.addParam("P$i", ln(bound.lower) * 1.01, ln(bound.upper) * 0.99)
        }
        return xSpace
    }

    /**
	 * @param data is a set of x (x:horizontal)
	 *
	 * @param covMat is the pre-computed covariance matrix
	 *
	 * @return |data| by |data| by |theta| covariance matrix between all pair of data's instances and its gradients
	 */
    override fun getCovarianceMatrixGradient(
        data: Matrix<Double>,
        covMat: Matrix<Double>,
        theta: DoubleArray
    ): NDArray<Double> {
        val (n, m) = data.shape()
        val grads = NDArray.doubleFactory.zeros(n, n, m)

        val lengthScale = getLengthScale(theta)

        for (i in 0 until n) {
            for (j in 0 until n) {
                for (k in 0 until m) {
                    grads[i, j, k] += (data[i, k] - data[j, k]).pow(2) / (lengthScale[k].pow(2)) * covMat[i, j]
                }
            }
        }

        return grads
    }

    /**
	 * @param x is a vertical vector
	 *
	 * @param y is a vertical vector
	 *
	 * @return K(x, y) = [ (x - y) .^ 2 ] dot theta is similarity between x and y
	 */
    private fun distance(x: Matrix<Double>, y: Matrix<Double>, lengthScale: DoubleArray): Double {
        val d = x - y
        return exp(-0.5 * (0 until d.numRows()).map { (d[it] / lengthScale[it]).pow(2) }.sum())
    }

    private fun getLengthScale(theta: DoubleArray): DoubleArray {
        return DoubleArray(theta.size) {
            val x = exp(theta[it])
            x
        }
    }
}
