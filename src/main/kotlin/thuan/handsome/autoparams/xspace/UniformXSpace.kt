package thuan.handsome.autoparams.xspace

import kotlin.math.roundToInt
import kotlin.random.Random

class UniformXSpace : XSpace {
	private val constantParams = mutableMapOf<String, Any>()
	private val isDoubleType = mutableListOf<Boolean>()
	private val lowerBounds = mutableListOf<Double>()
	private val upperBounds = mutableListOf<Double>()
	private val names = mutableListOf<String>()

	override fun addParam(name: String, lower: Double, upper: Double, isDouble: Boolean) {
		isDoubleType.add(isDouble)
		lowerBounds.add(lower)
		upperBounds.add(upper)
		names.add(name)
	}

	override fun addConstantParams(params: Map<String, Any>) {
		constantParams.putAll(params)
	}

	override fun sample(): Map<String, Any> {
		return constantParams + names.withIndex()
			.map { (index, name) ->
				name to if (isDoubleType[index])
					Random.nextDouble(
						lowerBounds[index],
						upperBounds[index]
					)
				else
					Random.nextInt(
						lowerBounds[index].roundToInt(),
						upperBounds[index].roundToInt() + 1
					)
			}
			.toMap()
	}

	override fun getDim(): Int {
		return names.size
	}
}