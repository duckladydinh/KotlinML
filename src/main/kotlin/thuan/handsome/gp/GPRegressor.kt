package thuan.handsome.gp

import koma.*
import koma.extensions.get
import koma.extensions.map
import koma.matrix.Matrix
import kotlin.math.PI
import kotlin.math.ln
import thuan.handsome.core.function.DifferentialEvaluation
import thuan.handsome.core.function.DifferentialFunction
import thuan.handsome.core.utils.toUncheckedDoubleArray
import thuan.handsome.core.xspace.XSpace
import thuan.handsome.gp.kernel.Kernel
import thuan.handsome.gp.kernel.RBFKernel
import thuan.handsome.optimizer.numeric.NumericOptimizer

class GPRegressor internal constructor(
    private val data: Matrix<Double>, // list of horizontal vectors
    private val y: Matrix<Double>, // vertical vector of size |data|
    private val kernel: Kernel = RBFKernel(data.numCols()),
    private val noise: Double = 1e-10,
    normalizeY: Boolean = false
) {
    var bestTheta = DoubleArray(data.numCols())

    private lateinit var covMatCholesky: Matrix<Double>
    private lateinit var covMatInv: Matrix<Double>
    private lateinit var covMat: Matrix<Double>
    // alpha is a vertical vector where covMat * alpha = y
    private lateinit var alpha: Matrix<Double>

    private var likelihoodGrads = DoubleArray(data.numCols())
    private var yMean = if (normalizeY) y.mean() else 0.0
    private var xSpace: XSpace = kernel.getThetaBounds()
    private var isPosterior = false
    private var likelihood = 0.0

    companion object {
        fun fit(data: Matrix<Double>, y: Matrix<Double>, numOptimizerRestarts: Int = 0): GPRegressor {
            val gp = GPRegressor(data, y)
            val func = DifferentialFunction {
                gp.evaluate(it, true)
            }

            var bestLogLikelihood = Double.NEGATIVE_INFINITY
            var bestTheta = gp.bestTheta

            for (iter in 0..numOptimizerRestarts) {
                val thetaZero = if (iter == 0) gp.bestTheta else gp.xSpace.sample().values.toUncheckedDoubleArray()
                val res = NumericOptimizer.maximize(func, thetaZero, bounds = gp.xSpace.getBounds())
                if (bestLogLikelihood < res.y) {
                    bestLogLikelihood = res.y
                    bestTheta = res.x
                }
            }

            // IMPORTANT: this line of code updates final theta, we
            // need a full gp for prediction so cannot only return
            // this. Hence we keep this as an internal state
            gp.bestTheta = bestTheta
            // we will not updated theta inside 'evaluate' to ensure
            // that bestTheta is really the best theta ever
            gp.evaluate(theta = bestTheta, computeGradient = true)
            return gp
        }
    }

    /**
	 * @param x a point which is the determining parameters of some function
	 *
	 * @return a mean and variance of x's goodness
	 */
    fun predict(x: DoubleArray): GPPrediction {
        require(x.size == xSpace.getDim())

        val xMat = create(x)
        var predYVar = kernel.getCovarianceMatrixTrace(xMat, xMat, this.bestTheta)[0]

        if (!this.isPosterior) {
            return GPPrediction(0.0, predYVar)
        }

        // horizontal vector
        val predK = kernel.getCovarianceMatrix(create(x), this.data, this.bestTheta)
        // definitely a double
        val predYMean = this.yMean + dot(predK, this.alpha)
        // horizontal vector predK * K
        val predKK = predK * this.covMatInv
        predYVar -= dot(predKK, predK)
        predYVar = max(0.0, predYVar)

        return GPPrediction(predYMean, predYVar)
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

        this.covMat = this.kernel.getCovarianceMatrix(this.data, theta) + eye(n) * noise
        this.covMatInv = this.covMat.inv()

        this.alpha = covMatInv * this.y
        this.covMatCholesky = covMat.chol()
        this.likelihood = 0.0

        this.likelihood -= covMatCholesky.diag().map { ln(it) }.elementSum()
        this.likelihood -= 0.5 * dot(this.y, alpha)
        this.likelihood -= 0.5 * n * ln(2 * PI)

        if (computeGradient) {
            val covMatGrads = this.kernel.getCovarianceMatrixGradient(this.data, this.covMat, theta)
            val tmp = (alpha * alpha.T) - this.covMatInv * eye(n)

            for (i in 0 until n) {
                for (j in 0 until n) {
                    for (k in 0 until m) {
                        this.likelihoodGrads[k] += tmp[i, j] * covMatGrads[i, j, k] / 2.0
                    }
                }
            }
        }

        this.isPosterior = true

        return DifferentialEvaluation(
            this.likelihood,
            this.likelihoodGrads
        )
    }
}
