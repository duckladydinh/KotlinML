package thuan.handsome.lbfgsb

/**
 * @param lower Value of the lower bound. No bound is defined if the value is null.
 * @param upper Value of the upper bound. No bound is defined if the value is null.
 */
data class Bound(val lower: Double?, val upper: Double?) {
	fun isLowerBoundDefined(): Boolean {
		return lower != null
	}

	fun isUpperBoundDefined(): Boolean {
		return upper != null
	}
}