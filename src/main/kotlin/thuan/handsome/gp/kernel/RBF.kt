package thuan.handsome.gp.kernel

import koma.extensions.*
import koma.matrix.Matrix
import koma.ndarray.NDArray
import koma.zeros
import kotlin.math.exp
import kotlin.math.ln
import thuan.handsome.core.xspace.*

class RBF constructor(private val bound: Bound = Bound(1e-5, 1e5)) : Kernel {
    companion object {
        /**
		 * @param dataX is a set of x (x:horizontal)
		 * @param dataY is a set of y (y:horizontal)
		 *
		 * @return |dataX| by |dataY| distance matrix between all pair of rows (x, y) and diagonal = 0...
		 */
        private fun getDists(
            dataX: Matrix<Double>,
            dataY: Matrix<Double>,
            theta: DoubleArray,
            symmetric: Boolean = false
        ): Matrix<Double> {
            val n = dataX.numRows()
            val m = dataY.numRows()
            val covMat = zeros(n, m)
            val lengthScale = getLengthScale(theta)

            if (symmetric) {
                for (i in 0 until n) {
                    for (j in i + 1 until m) {
                        val dist = distance(dataX.getRow(i).transpose(), dataY.getRow(j).transpose(), lengthScale)
                        covMat[i, j] = dist
                        covMat[j, i] = dist
                    }
                }
            } else {
                for (i in 0 until n) {
                    for (j in 0 until m) {
                        covMat[i, j] = distance(dataX.getRow(i).transpose(), dataY.getRow(j).transpose(), lengthScale)
                    }
                }
            }
            return covMat
        }

        /**
		 * @param x is a vertical vector
		 *
		 * @param y is a vertical vector
		 *
		 * @return d(x, y) = [ (x - y) ./ lengthScale .^ 2 ] is distance between x and y
		 */
        private fun distance(x: Matrix<Double>, y: Matrix<Double>, lengthScale: Double): Double {
            return ((x - y) / lengthScale).epow(2).elementSum()
        }

        /**
		 * @param dists = d(X, Y) from distance for the whole matrix
		 *
		 * @return K(x, y) = exp(-0.5 * d(x, y)) is the similarity between x and y
		 */
        private fun kernelApply(dists: Matrix<Double>): Matrix<Double> {
            return koma.exp(-0.5 * dists)
        }

        /**
		 * @param theta is external control parameter
		 *
		 * @return lengthScale, the primary control parameter of this kernel
		 */
        private fun getLengthScale(theta: DoubleArray): Double {
            require(theta.size == 1)
            return exp(theta[0])
        }
    }

    override fun getCovarianceMatrix(dataX: Matrix<Double>, dataY: Matrix<Double>, theta: DoubleArray): Matrix<Double> {
        val dists = getDists(dataX, dataY, theta)
        return kernelApply(dists)
    }

    override fun getCovarianceMatrixGradient(data: Matrix<Double>, theta: DoubleArray): NDArray<Double> {
        require(theta.size == getDim())
        val n = data.numRows()
        val m = getDim()

        val grads = NDArray.doubleFactory.zeros(n, n, m)
        val dists = getDists(data, data, theta)
        val covMat = kernelApply(dists)
        val tmp = covMat emul dists

        for (i in 0 until n) {
            for (j in 0 until n) {
                grads[i, j, 0] += tmp[i, j]
            }
        }

        return grads
    }

    override fun getCovarianceMatrixTrace(data: Matrix<Double>, theta: DoubleArray): DoubleArray {
        return DoubleArray(data.numRows()) { 1.0 }
    }

    override fun getCovarianceMatrix(data: Matrix<Double>, theta: DoubleArray): Matrix<Double> {
        val dists = getDists(data, data, theta, symmetric = true)
        return kernelApply(dists)
    }

    override fun getThetaBounds(): XSpace {
        val xSpace = UniformXSpace()
        xSpace.addParam("B", ln(bound.lower) * 1.01, ln(bound.upper) * 0.99)
        return xSpace
    }

    override fun getDim(): Int {
        return 1
    }
}
