package thuan.handsome.core.xspace

interface XSpace {
    fun addParam(name: String, lower: Double, upper: Double, xType: XType = XType.DOUBLE)

    fun addConstantParams(params: Map<String, Any>)

    fun decorate(x: DoubleArray): Map<String, Any>

    fun validate(index: Int, value: Any): Boolean

    fun getBounds(): Array<Bound>

    fun sample(): DoubleArray

    fun getDim(): Int
}
