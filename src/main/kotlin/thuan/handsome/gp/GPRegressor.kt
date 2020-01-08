package thuan.handsome.gp

import koma.*
import koma.extensions.get
import koma.extensions.map
import koma.matrix.Matrix
import kotlin.math.PI
import kotlin.math.ln
import thuan.handsome.core.function.DifferentialEvaluation
import thuan.handsome.core.function.DifferentialFunction
import thuan.handsome.core.utils.LOGGER
import thuan.handsome.core.xspace.XSpace
import thuan.handsome.gp.kernel.Kernel
import thuan.handsome.gp.kernel.RBF
import thuan.handsome.optimizer.numeric.NumericOptimizer

class GPRegressor internal constructor(
    private val data: Matrix<Double>, // list of horizontal vectors
    private val y: Matrix<Double>, // vertical vector of size |data|
    private val kernel: Kernel = RBF(),
    private val noise: Double = 1e-10,
    normalizeY: Boolean = false
) {
    var bestTheta = DoubleArray(kernel.getDim())

    private lateinit var covCholesky: Matrix<Double>
    private lateinit var covInv: Matrix<Double>
    private lateinit var cov: Matrix<Double>
    // alpha is a vertical vector where covMat * alpha = y
    private lateinit var alpha: Matrix<Double>

    private var likelihoodGrads = DoubleArray(kernel.getDim())
    private var yMean = if (normalizeY) y.mean() else 0.0
    private var xSpace: XSpace = kernel.getThetaBounds()
    private var isPosterior = false
    private var likelihood = 0.0

    companion object {
        fun fit(
            data: Matrix<Double>,
            y: Matrix<Double>,
            maxiter: Int = 1,
            kernel: Kernel = RBF(),
            noise: Double = 1e-10,
            normalizeY: Boolean = false
        ): GPRegressor {
            require(maxiter >= 1)

            val gp = GPRegressor(data, y, kernel, noise, normalizeY)
            val func = DifferentialFunction {
                gp.logLikelihood(it, true)
            }

            var bestLogLikelihood = Double.NEGATIVE_INFINITY
            var bestTheta = gp.bestTheta

            for (iter in 1..maxiter) {
                val thetaZero = if (iter == 0) gp.bestTheta else gp.xSpace.sample()
                val res = NumericOptimizer.maximize(func, thetaZero, bounds = gp.xSpace.getBounds())

                if (res.y > bestLogLikelihood) {
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
            gp.logLikelihood(theta = bestTheta, computeGradient = true)
            return gp
        }
    }

    /**
     * @param x a point which is the determining parameters of some function
     *
     * @return a mean and variance of x's goodness
     */
    fun predict(x: DoubleArray): GPPrediction {
        val xMat = create(x)
        var predYVar = kernel.getCovarianceMatrixTrace(xMat, this.bestTheta)[0]

        if (!this.isPosterior) {
            // this is the same prior probability for all kernels
            return GPPrediction(0.0, predYVar)
        }

        require(x.size == this.data.numCols())
        // predK is a horizontal vector and predYMean is definitely a double, we use dot product
        // here because we are certain that predK has only 1 row for 1 x
        val predK = kernel.getCovarianceMatrix(create(x), this.data, this.bestTheta)
        val predYMean = this.yMean + dot(predK, this.alpha)

        // horizontal vector predK * K
        val predKK = predK * this.covInv
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
    fun logLikelihood(theta: DoubleArray, computeGradient: Boolean = false): DifferentialEvaluation {
        require(theta.size == kernel.getDim())

        val n = this.data.numRows()
        val m = kernel.getDim()

        this.likelihood = Double.NEGATIVE_INFINITY
        for (i in 0 until m) {
            this.likelihoodGrads[i] = 0.0
        }
        for (i in 0 until m) {
            if (!xSpace.validate(i, theta[i])) {
                return DifferentialEvaluation(this.likelihood, this.likelihoodGrads)
            }
        }

        this.cov = this.kernel.getCovarianceMatrix(this.data, theta) + eye(n) * noise
        this.covInv = this.cov.inv()

        this.alpha = this.covInv * this.y
        try {
            this.covCholesky = cov.chol()
        } catch (e: IllegalStateException) {
            LOGGER.warn { e }
            return DifferentialEvaluation(this.likelihood, this.likelihoodGrads)
        }

        // 1. Full formula:
        // log[p(y|X, theta)] =
        //      - 0.5 * log(K_theta + var * I)
        //      - 0.5 * (y - m_theta).T * (K_theta + var * I) * (y - m_theta)
        //      - 0.5 * n * log(2 * PI)
        // 2. Dot product is used since we know they are vectors and we want
        // the final value to be a scalar. Alternatively, we could write:
        //      (this.y.T * alpha) [0, 0]
        this.likelihood =
            -covCholesky.diag().map { ln(it) }.elementSum() - 0.5 * dot(this.y, this.alpha) - 0.5 * n * ln(2 * PI)

        if (computeGradient) {
            // Full formula: grad[i] = 0.5 * tr( (alpha * alpha.T - K.inv()) * kernel_gradient_for grad[i])
            val covGrads = this.kernel.getCovarianceMatrixGradient(this.data, theta)
            val tmp = ((alpha * alpha.T) - this.covInv).T

            for (i in 0 until n) {
                for (j in 0 until n) {
                    for (k in 0 until m) {
                        // this is to avoid matrix multiplication since only trace is needed (tmp has been transposed)
                        this.likelihoodGrads[k] += tmp[i, j] * covGrads[i, j, k] / 2.0
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
