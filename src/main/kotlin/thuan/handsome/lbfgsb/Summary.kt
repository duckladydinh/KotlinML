package thuan.handsome.lbfgsb

data class Summary(
    val x: DoubleArray,
    val y: Double,
    val grads: DoubleArray,
    val numIterations: Int,
    val numEvals: Int,
    val type: StopType,
    val stateDescription: String
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Summary

        if (!x.contentEquals(other.x)) return false
        if (y != other.y) return false
        if (!grads.contentEquals(other.grads)) return false
        if (numIterations != other.numIterations) return false
        if (numEvals != other.numEvals) return false
        if (type != other.type) return false
        if (stateDescription != other.stateDescription) return false

        return true
    }

    override fun hashCode(): Int {
        var result = x.contentHashCode()
        result = 31 * result + y.hashCode()
        result = 31 * result + grads.contentHashCode()
        result = 31 * result + numIterations
        result = 31 * result + numEvals
        result = 31 * result + type.hashCode()
        result = 31 * result + stateDescription.hashCode()
        return result
    }
}
