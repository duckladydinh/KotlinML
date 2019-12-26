package thuan.handsome.autoparams.optimizer

import thuan.handsome.ml.utils.LOGGER
import thuan.handsome.ml.xspace.XSpace

class UniformRandomOptimizer : Optimizer {
    override fun argMaximize(
        func: (Map<String, Any>) -> Double,
        xSpace: XSpace,
        maxEvals: Int
    ): Pair<Map<String, Any>, Double> {
        var bestX = mapOf<String, Any>()
        var bestY = Double.NEGATIVE_INFINITY

        for (iter in 1..maxEvals) {
            val x = xSpace.sample()
            val y = func.invoke(x)

            if (bestY < y) {
                bestY = y
                bestX = x
            }

            LOGGER.info { "Iteration %3d | y = %.6f | bestY = %.6f.".format(iter, y, bestY) }
        }

        return Pair(bestX, bestY)
    }
}
