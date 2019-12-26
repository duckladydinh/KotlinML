package thuan.handsome.core.xspace

interface XSpace {
    fun addParam(name: String, lower: Double, upper: Double, isDouble: Boolean = true)

    fun addConstantParams(params: Map<String, Any>)

    fun validate(index: Int, value: Any): Boolean

    fun sample(): Map<String, Any>

    fun getBounds(): Array<Bound>

    fun getDim(): Int
}
