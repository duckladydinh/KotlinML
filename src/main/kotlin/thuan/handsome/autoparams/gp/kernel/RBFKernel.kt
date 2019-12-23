package thuan.handsome.autoparams.gp.kernel

import koma.*
import koma.extensions.*
import koma.matrix.Matrix
import kotlin.random.Random

class RBFKernel(private var theta: DoubleArray) : Kernel {
    companion object {
        private const val SEARCH_UPPER_BOUND = 50
    }

    constructor(dimensions: Int) : this(DoubleArray(dimensions) {
        Random.nextDouble() * SEARCH_UPPER_BOUND
    })

    override fun evaluate(x: Matrix<Double>, y: Matrix<Double>): Double {
        assert(x.numRows() == y.numRows() && x.numCols() == 1 && y.numCols() == 1)

        val params = create(theta)
        val diff = x - y

        val res = dot(diff emul diff, params)
        return exp(-0.5 * res)
    }

    override fun getCovarianceMatrix(X: Matrix<Double>): Matrix<Double> {
        val n = X.numRows()
        val covMat = zeros(n, n)

        for (i in 0 until n) {
            for (j in i until n) {
                covMat[i, j] = evaluate(X.getRow(i).transpose(), X.getRow(j).transpose())
                covMat[j, i] = covMat[i, j]
            }
        }
        return covMat
    }

    override fun getParameters(): DoubleArray {
        return theta
    }

    override fun updateParameters(params: DoubleArray) {
        this.theta = params
    }

    override fun getDimensions(): Int {
        return theta.size
    }
}
