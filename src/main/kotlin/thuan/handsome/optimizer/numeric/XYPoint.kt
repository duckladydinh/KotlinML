package thuan.handsome.optimizer.numeric

data class XYPoint(val x: DoubleArray, val y: Double) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as XYPoint

        if (!x.contentEquals(other.x)) return false
        if (y != other.y) return false

        return true
    }

    override fun hashCode(): Int {
        var result = x.contentHashCode()
        result = 31 * result + y.hashCode()
        return result
    }
}
