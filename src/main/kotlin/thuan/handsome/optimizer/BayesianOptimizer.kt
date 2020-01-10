package thuan.handsome.optimizer

import koma.create
import kotlin.math.sqrt
import thuan.handsome.core.function.DifferentialFunction
import thuan.handsome.core.utils.LOGGER
import thuan.handsome.core.xspace.XSpace
import thuan.handsome.gp.GPRegressor
import thuan.handsome.gp.kernel.Kernel
import thuan.handsome.gp.kernel.Matern
import thuan.handsome.optimizer.numeric.NumericOptimizer
import thuan.handsome.optimizer.numeric.XYPoint

class BayesianOptimizer(val kernel: Kernel = Matern()) : Optimizer {
    private companion object {
        fun suggest(xSpace: XSpace, func: (DoubleArray) -> Double, maxiter: Int = 1000): DoubleArray {
            var bestX = xSpace.sample()
            var bestValue = func.invoke(bestX)
            val diffFunc = DifferentialFunction.from { func.invoke(it) }

            for (iter in 2..maxiter) {
                val xZero = xSpace.sample()
                val summary = NumericOptimizer.maximize(
                    diffFunc,
                    xZero,
                    xSpace.getBounds()
                )
                if (summary.y > bestValue) {
                    bestValue = summary.y
                    bestX = summary.x
                }
            }

            LOGGER.debug { "Next x's UCP Score = $bestValue" }
            return bestX
        }

        fun ucb(x: DoubleArray, gp: GPRegressor, kappa: Double = 1.4142135623730951): Double {
            val (mean, variance) = gp.predict(x)
            val std = sqrt(variance)
            return mean + kappa * std
        }
    }
    override fun argmax(
        func: (Map<String, Any>) -> Double,
        xSpace: XSpace,
        maxiter: Int
    ): Pair<Map<String, Any>, Double> {
        val f = fun(x: DoubleArray): Double {
            val params = xSpace.decorate(x)
            return func.invoke(params)
        }

        val (x, y) = maximize(f, xSpace, maxiter)
        return Pair(xSpace.decorate(x), y)
    }

    private fun maximize(func: (DoubleArray) -> Double, xSpace: XSpace, maxiter: Int = 30): XYPoint {
        require(maxiter >= 5)

        var bestX = xSpace.sample()
        var bestY = func.invoke(bestX)

        val data = mutableListOf(bestX)
        val y = mutableListOf(bestY)

        LOGGER.debug { "WARMING UP" }
        LOGGER.debug { "Iteration %3d | y = %.6f | bestY = %.6f.".format(1, bestY, bestY) }

        // warm-up
        for (iter in 2..5) {
            val x = xSpace.sample()
            val res = func.invoke(x)
            data.add(x)
            y.add(res)

            if (res > bestY) {
                bestY = res
                bestX = x
            }
            LOGGER.debug { "Iteration %3d | y = %.6f | bestY = %.6f.".format(iter, res, bestY) }
        }

        LOGGER.debug { "START TRAINING" }
        for (iter in 6..maxiter) {
            val dataMat = create(data.toTypedArray())
            val yMat = create(y.toDoubleArray()).T

            val gp = GPRegressor.fit(dataMat, yMat, 5, kernel, 1e-6, true)
            val ac = fun(x: DoubleArray): Double {
                return ucb(x, gp)
            }

            val x = suggest(xSpace, ac)
            val res = func.invoke(x)

            if (res > bestY) {
                bestY = res
                bestX = x
            }

            data.add(x)
            y.add(res)
            LOGGER.debug { "Iteration %3d | y = %.6f | bestY = %.6f.".format(iter, res, bestY) }
        }

        return XYPoint(bestX, bestY)
    }
}
