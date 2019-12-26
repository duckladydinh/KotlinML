package thuan.handsome.core.xspace

import kotlin.math.ceil
import kotlin.random.Random

class UniformXSpace : XSpace {
    private val constantParams = mutableMapOf<String, Any>()
    private val paramType = mutableListOf<XType>()
    private val bounds = mutableListOf<Bound>()
    private val names = mutableListOf<String>()

    override fun addParam(name: String, lower: Double, upper: Double, isDouble: Boolean) {
        bounds.add(Bound(lower, upper))
        paramType.add(if (isDouble) XType.DOUBLE else XType.INT)
        names.add(name)
    }

    override fun addConstantParams(params: Map<String, Any>) {
        constantParams.putAll(params)
    }

    override fun sample(): Map<String, Any> {
        return constantParams + names.withIndex()
            .map { (index, name) ->
                val value: Any = when (paramType[index]) {
                    XType.DOUBLE -> Random.nextDouble(
                        bounds[index].lower,
                        bounds[index].upper
                    )
                    XType.INT -> Random.nextInt(
                        ceil(bounds[index].lower).toInt(),
                        bounds[index].upper.toInt() + 1
                    )
                }
                name to value
            }
            .toMap()
    }

    override fun validate(index: Int, value: Any): Boolean {
        val x = (value as Number).toDouble()
        return x >= bounds[index].lower && x <= bounds[index].upper
    }

    override fun getBounds(): Array<Bound> {
        return bounds.toTypedArray()
    }

    override fun getDim(): Int {
        return names.size
    }
}
