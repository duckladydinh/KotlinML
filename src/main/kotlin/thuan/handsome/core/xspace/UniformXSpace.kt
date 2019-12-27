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

    override fun decorate(x: DoubleArray): Map<String, Any> {
        require(this.getDim() == x.size)
        val params = constantParams.toMutableMap()
        for ((i, name) in names.withIndex()) {
            if (paramTypes[i] == XType.INT) {
                params[name] = x[i].toUncheckedInt()
            } else {
                params[name] = x[i]
            }
        }
        return params
    }

    override fun validate(index: Int, value: Any): Boolean {
        val x = value.toUncheckedDouble()
        return x >= bounds[index].lower && x <= bounds[index].upper
    }

    override fun getBounds(): Array<Bound> {
        return bounds.toTypedArray()
    }

    override fun sample(): DoubleArray {
        return bounds.map(Bound::sample).toDoubleArray()
    }

    override fun getDim(): Int {
        return names.size
    }
}
