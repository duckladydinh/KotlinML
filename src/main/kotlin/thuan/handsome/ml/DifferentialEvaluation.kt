package thuan.handsome.ml

data class DifferentialEvaluation(val y: Double, val grads: DoubleArray) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DifferentialEvaluation

        if (y != other.y) return false
        if (!grads.contentEquals(other.grads)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = y.hashCode()
        result = 31 * result + grads.contentHashCode()
        return result
    }
}
