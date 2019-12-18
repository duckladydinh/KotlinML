package thuan.handsome.autoparams

import thuan.handsome.utils.LOGGER
import kotlin.random.Random
import kotlin.test.assertTrue

data class Param(val name: String, val lowerBound: Any, val upperBound: Any, val isInt: Boolean = false) {
	var value: Any = if (!isInt) Random.nextDouble(lowerBound as Double, upperBound as Double)
	else Random.nextInt(lowerBound as Int, upperBound as Int)
}

fun maximize(
	objFunc: (Map<String, Any>) -> Double,
	params: List<Param>,
	maxEvals: Int
): Pair<Map<String, Any>, Double> {
	assertTrue(maxEvals > 0, "At least 1 evaluation is needed!")

	var bestY = Double.NEGATIVE_INFINITY
	lateinit var bestX: Map<String, Any>

	for (iter in 1..maxEvals) {
		val parameters = params.map {
			it.name to if (!it.isInt) Random.nextDouble(
				it.lowerBound as Double,
				it.upperBound as Double
			) else Random.nextInt(
				it.lowerBound as Int,
				it.upperBound as Int
			)
		}.toMap()

		val res = objFunc.invoke(parameters)
		if (res > bestY) {
			bestY = res
			bestX = parameters
		}
		LOGGER.info("Iter $iter | Score = $res | Best = $bestY")
	}

	return Pair(bestX, bestY)
}