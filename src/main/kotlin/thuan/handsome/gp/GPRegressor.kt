package thuan.handsome.gp

import koma.*
import koma.extensions.*
import koma.matrix.Matrix
import kotlin.math.PI
import kotlin.math.ln
import thuan.handsome.gp.kernel.Kernel
import thuan.handsome.gp.kernel.RBFKernel

class GPRegressor internal constructor(
    private val data: Matrix<Double>, // list of horizontal vectors
    private val y: Matrix<Double>, // vertical vector of size |data|
    private val kernel: Kernel = RBFKernel(data.numCols()),
    private val noise: Double = 1e-10
) {
    private lateinit var covMatInv: Matrix<Double>
    private lateinit var covMat: Matrix<Double>

    var logMarginalLikelihoodGrads = DoubleArray(data.numCols())
    var theta = DoubleArray(data.numCols())
    var logMarginalLikelihood = 0.0
    var variance = 0.0
    var mean = 0.0

    companion object {
        fun fit(data: Matrix<Double>, y: Matrix<Double>, maxiter: Int = 1): GPRegressor {
            val gp = GPRegressor(data, y)

            var maxLogLikelihood = Double.NEGATIVE_INFINITY
            val func = { theta: DoubleArray ->
                gp.updateTheta(theta)

                -gp.logMarginalLikelihood
            }

            for (iter in 1..maxiter) {
                val thetaZero = (rand(data.numCols(), 1) * 50.0).toDoubleArray()
            }

            return gp
        }
    }

    fun predict(x: DoubleArray): Map<String, Double> {
        require(x.size == kernel.dimension)

        val predCovMat = kernel.getCovarianceMatrix(create(x), this.data, this.theta)

        val predMean = this.mean + dot(covMat.T * this.covMatInv, this.y - this.mean)

        val tmp = this.covMatInv * predCovMat
        val predVariance = max(
            0.0,
            this.variance * (1 - dot(predCovMat.T, tmp) + (1 - tmp.elementSum()).pow(2) / this.covMatInv.elementSum())
        )

        return mapOf("mean" to predMean, "std" to sqrt(predVariance))
    }

    /**
	 * Since kernel.theta can change in each iteration,
	 * this method will give different values accordingly
	 *
	 * likelihood = P(y | X, theta)
	 */
    fun updateTheta(theta: DoubleArray, computeGradient: Boolean = false) {
        require(this.kernel.dimension == theta.size)

        val invalid = theta.any { it < kernel.lowerBound || it > kernel.upperBound }
        if (invalid) {
            this.logMarginalLikelihood = Double.NEGATIVE_INFINITY
            return
        }

        this.theta = theta

        val (n, m) = this.data.shape()

        this.covMat = this.kernel.getCovarianceMatrix(this.data, this.theta) + eye(n) * noise
        this.covMatInv = this.covMat.inv()

        val alpha = covMatInv * this.y
        val cholesky = covMat.chol()
        this.logMarginalLikelihood = 0.0

        this.logMarginalLikelihood -= cholesky.diag().map { ln(it) }.elementSum()
        this.logMarginalLikelihood -= 0.5 * dot(this.y, alpha)
        this.logMarginalLikelihood -= 0.5 * n * ln(2 * PI)

        if (computeGradient) {
            val covMatGrads = this.kernel.getCovarianceMatrixGradient(this.data, this.covMat, this.theta)
            val tmp = (alpha * alpha.T) - this.covMatInv * eye(n)

            repeat(m) {
                this.logMarginalLikelihoodGrads[it] = 0.0
            }
            for (i in 0 until n) {
                for (j in 0 until n) {
                    for (k in 0 until m) {
                        this.logMarginalLikelihoodGrads[k] += tmp[i, j] * covMatGrads[i, j, k] / 2.0
                    }
                }
            }
        }
    }
}
