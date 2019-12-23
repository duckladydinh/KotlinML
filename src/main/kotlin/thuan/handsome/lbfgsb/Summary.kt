package thuan.handsome.lbfgsb

class Summary(
	val x: DoubleArray,
	val y: Double,
	val grads: DoubleArray,
	val stopInfo: StopInfo
) {
	override fun toString(): String {
		return "Summary = {\n" +
				"\tx       : ${toString(x)}\n" +
				"\ty       : $y\n" +
				"\tgrads   : ${toString(grads)}\n" +
				"\tstopInfo: $stopInfo\n" +
				"}"
	}

	companion object {
		private fun toString(values: DoubleArray): String {
			val description = values.joinToString ("," )
			return "[$description]"
		}
	}
}