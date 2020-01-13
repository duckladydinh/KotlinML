package thuan.handsome.core.optimizer

import thuan.handsome.core.utils.Logger
import thuan.handsome.core.xspace.XSpace

class RandomOptimizer : Optimizer {
    override fun argmax(
        func: (Map<String, Any>) -> Double,
        xSpace: XSpace,
        maxiter: Int
    ): Pair<Map<String, Any>, Double> {
        var bestX = mapOf<String, Any>()
        var bestY = Double.NEGATIVE_INFINITY

        for (iter in 1..maxiter) {
            val x = xSpace.decorate(xSpace.sample())
            val y = func.invoke(x)
            if (bestY < y) {
                bestY = y
                bestX = x
            }
            Logger.fine().log("Iteration %3d | y = %.6f | bestY = %.6f.".format(iter, y, bestY))
        }
        return Pair(bestX, bestY)
    }
}
