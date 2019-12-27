package thuan.handsome.core.xspace

import thuan.handsome.core.utils.toUncheckedDouble
import thuan.handsome.core.utils.toUncheckedInt

class UniformXSpace : XSpace {
    private val constantParams = mutableMapOf<String, Any>()
    private val paramTypes = mutableListOf<XType>()
    private val bounds = mutableListOf<Bound>()
    private val names = mutableListOf<String>()

    override fun addParam(name: String, lower: Double, upper: Double, xType: XType) {
        bounds.add(Bound(lower, upper))
        paramTypes.add(xType)
        names.add(name)
    }

    override fun addConstantParams(params: Map<String, Any>) {
        constantParams.putAll(params)
    }

    override fun validate(index: Int, value: Any): Boolean {
        val x = value.toUncheckedDouble()
        return x >= bounds[index].lower && x <= bounds[index].upper
    }

    override fun sampleWithConstants(): Map<String, Any> {
        val variableParams = this.sample()
        val params = (constantParams + variableParams).toMutableMap()
        (names zip paramTypes).filter { it.second == XType.INT }.forEach {
            params[it.first] = variableParams[it.first].toUncheckedInt()
        }
        return params
    }

    override fun sample(): Map<String, Double> {
        return (names zip bounds)
            .map { (name, bound) -> name to bound.sample() }
            .toMap()
    }

    override fun getBounds(): Array<Bound> {
        return bounds.toTypedArray()
    }

    override fun getDim(): Int {
        return names.size
    }
}
