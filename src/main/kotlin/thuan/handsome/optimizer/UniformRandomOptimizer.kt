package thuan.handsome.optimizer

import thuan.handsome.core.utils.LOGGER
import thuan.handsome.core.xspace.XSpace

class UniformRandomOptimizer : Optimizer {
    override fun argmax(
        func: (Map<String, Any>) -> Double,
        xSpace: XSpace,
        maxiter: Int
    ): Pair<Map<String, Any>, Double> {
        var bestX = mapOf<String, Any>()
        var bestY = Double.NEGATIVE_INFINITY

        for (iter in 1..maxiter) {
            val x = xSpace.sampleWithConstants()
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
