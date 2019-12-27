package thuan.handsome.core.xspace

interface XSpace {
    fun addParam(name: String, lower: Double, upper: Double, xType: XType = XType.DOUBLE)

    fun addConstantParams(params: Map<String, Any>)

    fun validate(index: Int, value: Any): Boolean

    fun sampleWithConstants(): Map<String, Any>

    fun sample(): Map<String, Double>

    fun getBounds(): Array<Bound>

    fun getDim(): Int
}
