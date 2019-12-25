package thuan.handsome.lbfgsb

data class Bound(val lower: Double = Double.NEGATIVE_INFINITY, val upper: Double = Double.POSITIVE_INFINITY) {
    fun isLowerBounded(): Boolean {
        return lower > Double.NEGATIVE_INFINITY
    }

    fun isUpperBounded(): Boolean {
        return upper < Double.POSITIVE_INFINITY
    }

    fun times(n: Int): Array<Bound> {
        return Array(n) {
            this.copy()
        }
    }
}
