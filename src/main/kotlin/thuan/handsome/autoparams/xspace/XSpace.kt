package thuan.handsome.autoparams.xspace

interface XSpace {
    fun addParam(name: String, lower: Double, upper: Double, isDouble: Boolean = true)

    fun addConstantParams(params: Map<String, Any>)

    fun sample(): Map<String, Any>

    fun getDim(): Int
}
