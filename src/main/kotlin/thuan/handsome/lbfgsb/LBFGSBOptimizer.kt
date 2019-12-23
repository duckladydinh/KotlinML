package thuan.handsome.lbfgsb

/**
 * High-level wrapper for the L-BFGS-B algorithm
 */
class LBFGSBOptimizer {
    companion object {
        private const val DEFAULT_FUNCTION_REDUCTION_FACTOR = 1e7
        private const val DEFAULT_NUMBER_OF_CORRECTIONS = 5
        private const val DEFAULT_MAX_GRADIENT_NORM = 1e-5
        private const val DEFAULT_MAX_ITERATION = 100

        /**
		 * @param anyFunc a differential function
		 *
		 * @param maxGradientNorm maximal acceptable gradient value
		 * The iteration will stop when $||proj g||_{\infty}  value$
		 * where $proj g$ is the projected gradient.
		 *
		 * @param numCorrections Set number of corrections used in the limited memory matrix.
		 * According to the original fortran documentation,
		 * [3, 20] range is recommended.
		 *
		 * @param functionReductionFactor relative function reduction factor.
		 * The iteration will stop when
		 * (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1}  factr*epsmch
		 * where epsmch is the machine precision which is automatically
		 * generated by the code.
		 *
		 * Example values for 15 digits accuracy:
		 *  * 1e+12 for low accuracy,
		 *  * 1e+7  for moderate accuracy,
		 *  * 1e+1  for extremely high accuracy
		 */

        fun minimize(
            anyFunc: (DoubleArray) -> Double,
            xZero: DoubleArray,
            bounds: List<Bound> = getUnbounds(xZero.size),
            maxIterations: Int = DEFAULT_MAX_ITERATION,
            numCorrections: Int = DEFAULT_NUMBER_OF_CORRECTIONS,
            maxGradientNorm: Double = DEFAULT_MAX_GRADIENT_NORM,
            functionReductionFactor: Double = DEFAULT_FUNCTION_REDUCTION_FACTOR,
            onIterationFinished: ((DoubleArray, Double, DoubleArray) -> Boolean)? = null,

            debugLevel: Int = -1
        ): Summary {
            val func = differentialFunctionOf(anyFunc)
            assert(bounds.size == xZero.size) {
                "Bounds number (${bounds.size}) doesn't match starting point size (${xZero.size})"
            }

            val optimizer = LBFGSBWrapper(xZero.size, numCorrections).apply {
                setX(xZero)
                setBounds(bounds)
                setDebugLevel(debugLevel)
                setMaxGradientNorm(maxGradientNorm)
                setFunctionFactor(functionReductionFactor)
            }

            val info = optimizer.minimize(func, maxIterations, onIterationFinished)
            val summary = Summary(optimizer.getX(), optimizer.getY(), optimizer.getGrads(), info)

            optimizer.close()
            return summary
        }

        private fun getUnbounds(numVariables: Int) = (1..numVariables).map { Bound(null, null) }.toList()

        private fun differentialFunctionOf(func: (DoubleArray) -> Double): (DoubleArray) -> Pair<Double, DoubleArray> {
            return { x ->
                val y = func.invoke(x)
                val grads = gradientsOf(func, x, y, 1e-6)
                Pair(y, grads)
            }
        }

        private fun gradientsOf(
            func: (DoubleArray) -> Double,
            xZero: DoubleArray,
            yZero: Double,
            @Suppress("SameParameterValue") epsilon: Double
        ): DoubleArray {
            assert(epsilon > 0)

            return DoubleArray(xZero.size) {
                xZero[it] += epsilon
                val y = func.invoke(xZero)
                val gradient = (y - yZero) / epsilon
                xZero[it] -= epsilon

                gradient
            }
        }
    }
}