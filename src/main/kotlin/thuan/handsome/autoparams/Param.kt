package thuan.handsome.autoparams

import kotlin.random.Random
import kotlin.test.assertTrue

data class Param(val name: String, val lowerBound: Any, val upperBound: Any, val isInt: Boolean = false) {
	var value: Any = if (!isInt) Random.nextDouble(lowerBound as Double, upperBound as Double)
	else Random.nextInt(lowerBound as Int, upperBound as Int)
}

fun makeParams(params: List<Param>): Map<String, Any> {
	return params.map { it.name to it.value }.toMap()
}

fun maximize(
	objFunc: (MutableMap<String, Any>) -> Double,
	params: List<Param>,
	maxEval: Int
): Pair<MutableMap<String, Any>, Double> {
	assertTrue(maxEval > 0, "At least 1 evaluation is needed!")

	var bestY = Double.NEGATIVE_INFINITY
	lateinit var bestX: Map<String, Any>

	repeat(maxEval) {
		val paramsMap = params.map {
			it.name to if (!it.isInt) Random.nextDouble(
				it.lowerBound as Double,
				it.upperBound as Double
			) else Random.nextInt(
				it.lowerBound as Int,
				it.upperBound as Int
			)
		}.toMap().toMutableMap()
		val res = objFunc.invoke(paramsMap)
		if (res > bestY) {
			bestY = res
			bestX = paramsMap
		}
		println("Iter ${it + 1} = $res | Best = $bestY")
	}

	return Pair(bestX.toMutableMap(), bestY)
}