package thuan.handsome.gp.kernel

import koma.extensions.*
import koma.matrix.Matrix
import koma.ndarray.NDArray
import kotlin.math.exp
import kotlin.math.sqrt
import thuan.handsome.core.xspace.Bound
import thuan.handsome.gp.kernel.MaternType.*

class Matern constructor(private val nu: MaternType = TWICE_DIFFERENTIAL, bound: Bound = Bound(1e-5, 1e5)) : RBF(bound) {
    companion object {
        val ROOT_THREE = sqrt(3.0)
        val ROOT_FIVE = sqrt(5.0)
    }

    override fun getCovarianceMatrix(dataX: Matrix<Double>, dataY: Matrix<Double>, theta: DoubleArray): Matrix<Double> {
        val dists = koma.sqrt(getDists(dataX, dataY, theta))
        return kernelApply(dists)
    }

    override fun getCovarianceMatrixGradient(data: Matrix<Double>, theta: DoubleArray): NDArray<Double> {
        require(theta.size == getDim())
        val n = data.numRows()
        val m = getDim()

        val grads = NDArray.doubleFactory.zeros(n, n, m)
        val dists = getDists(data, data, theta, symmetric = true)
        val covMat = kernelApply(koma.sqrt(dists))

        when (nu) {
            ABSOLUTE_EXPONENTIAL -> {
                for (i in 0 until n) {
                    for (j in 0 until n) {
                        val g = covMat[i, j] * sqrt(dists[i, j])
                        if (g >= -1e9 && g <= 1e9) {
                            grads[i, j, 0] = g
                        }
                    }
                }
            }
            ONCE_DIFFERENTIAL -> {
                for (i in 0 until n) {
                    for (j in 0 until n) {
                        val g = 3 * dists[i, j] * exp(-sqrt(3 * dists[i, j]))
                        grads[i, j, 0] = g
                    }
                }
            }
            TWICE_DIFFERENTIAL -> {
                for (i in 0 until n) {
                    for (j in 0 until n) {
                        val tmp = sqrt(5 * dists[i, j])
                        val g = 5.0 / 3 * dists[i, j] * (tmp + 1) * exp(-tmp)
                        grads[i, j, 0] = g
                    }
                }
            }
        }

        return grads
    }

    override fun getCovarianceMatrix(data: Matrix<Double>, theta: DoubleArray): Matrix<Double> {
        val dists = koma.sqrt(getDists(data, data, theta))
        return kernelApply(dists)
    }

    /**
     * @param dists = d(X, Y) from distance for the whole matrix
     *
     * @return K(x, y) = exp(-0.5 * d(x, y)) is the similarity between x and y
     */
    private fun kernelApply(dists: Matrix<Double>): Matrix<Double> {
        return when (nu) {
            ABSOLUTE_EXPONENTIAL -> { return koma.exp(-dists) }
            ONCE_DIFFERENTIAL -> {
                val tmp = dists.times(ROOT_THREE)
                val covMat = tmp.plus(1.0).emul(koma.exp(-tmp))
                covMat
            }
            TWICE_DIFFERENTIAL -> {
                val tmp = dists.times(ROOT_FIVE)
                val covMat = (tmp.plus(1) + tmp.epow(2).div(3.0)).emul(koma.exp(-tmp))
                covMat
            }
        }
    }
}
