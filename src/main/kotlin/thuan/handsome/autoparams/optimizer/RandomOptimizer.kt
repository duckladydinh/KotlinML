package thuan.handsome.autoparams.optimizer

import thuan.handsome.autoparams.optimizer.Optimizer
import thuan.handsome.autoparams.xspace.XSpace

class RandomOptimizer : Optimizer {
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
		}

		return Pair(bestX, bestY)
	}
}