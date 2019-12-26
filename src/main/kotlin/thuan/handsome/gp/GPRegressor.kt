package thuan.handsome.gp

import koma.*
import koma.extensions.get
import koma.extensions.map
import koma.matrix.Matrix
import kotlin.math.PI
import kotlin.math.ln
import thuan.handsome.core.function.DifferentialEvaluation
import thuan.handsome.core.function.DifferentialFunction
import thuan.handsome.core.xspace.XSpace
import thuan.handsome.gp.kernel.Kernel
import thuan.handsome.gp.kernel.RBFKernel
import thuan.handsome.lbfgsb.CWrapper

class GPRegressor internal constructor(
    private val data: Matrix<Double>, // list of horizontal vectors
    private val y: Matrix<Double>, // vertical vector of size |data|
    private val kernel: Kernel = RBFKernel(data.numCols()),
    private val noise: Double = 1e-10,
    normalizeY: Boolean = false
) {
    var theta = DoubleArray(data.numCols())

    private lateinit var covMatInv: Matrix<Double>
    private lateinit var covMat: Matrix<Double>

    private var likelihoodGrads = DoubleArray(data.numCols())
    private var xSpace: XSpace = kernel.getThetaBounds()
    private var likelihood = 0.0
    private var variance = 0.0
    private var mean = if (normalizeY) y.mean() else 0.0

    companion object {
        fun fit(data: Matrix<Double>, y: Matrix<Double>, numOptimizerRestarts: Int = 0): GPRegressor {
            val gp = GPRegressor(data, y)

            var bestLogLikelihood = Double.NEGATIVE_INFINITY
            var bestTheta = gp.theta
            val func = DifferentialFunction { theta ->
                val (y, grads) = gp.evaluate(theta, true)
                DifferentialEvaluation(-y, grads.map { -it }.toDoubleArray())
            }

            val result = CWrapper.minimize(func, gp.theta, bounds = gp.xSpace.getBounds())
            if (bestLogLikelihood < -result.y) {
                bestLogLikelihood = -result.y
                bestTheta = result.x
            }

            repeat(numOptimizerRestarts) {
                val thetaZero = gp.xSpace.sample().values.map { (it as Number).toDouble() }.toDoubleArray()
                val result = CWrapper.minimize(func, thetaZero, bounds = gp.xSpace.getBounds())
                if (bestLogLikelihood < -result.y) {
                    bestLogLikelihood = -result.y
                    bestTheta = result.x
                }
            }

            gp.theta = bestTheta
            return gp
        }
    }

    fun predict(x: DoubleArray): Map<String, Double> {
        require(x.size == xSpace.getDim())

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

    fun evaluate(theta: DoubleArray, computeGradient: Boolean = false): DifferentialEvaluation {
        val (n, m) = this.data.shape()
        require(m == xSpace.getDim())
        this.theta = theta

        this.likelihood = Double.NEGATIVE_INFINITY
        repeat(m) {
            this.likelihoodGrads[it] = 0.0
        }
        val valid = (0 until m).all {
            xSpace.validate(it, theta[it])
        }
        if (!valid) {
            return DifferentialEvaluation(
                this.likelihood,
                this.likelihoodGrads
            )
        }

        this.covMat = this.kernel.getCovarianceMatrix(this.data, this.theta) + eye(n) * noise
        this.covMatInv = this.covMat.inv()

        val alpha = covMatInv * this.y
        val cholesky = covMat.chol()
        this.likelihood = 0.0

        this.likelihood -= cholesky.diag().map { ln(it) }.elementSum()
        this.likelihood -= 0.5 * dot(this.y, alpha)
        this.likelihood -= 0.5 * n * ln(2 * PI)

        if (computeGradient) {
            val covMatGrads = this.kernel.getCovarianceMatrixGradient(this.data, this.covMat, this.theta)
            val tmp = (alpha * alpha.T) - this.covMatInv * eye(n)

            for (i in 0 until n) {
                for (j in 0 until n) {
                    for (k in 0 until m) {
                        this.likelihoodGrads[k] += tmp[i, j] * covMatGrads[i, j, k] / 2.0
                    }
                }
            }
        }

        return DifferentialEvaluation(
            this.likelihood,
            this.likelihoodGrads
        )
    }
}
