package thuan.handsome.core.metrics

import kotlin.math.roundToInt

class F1Score : Metric {
    override fun evaluate(predicted: DoubleArray, target: DoubleArray): Double {
        require(predicted.size == target.size)
        require(predicted.isNotEmpty())

        val n = predicted.size
        var a = 0
        var b = 0
        repeat(n) {
            val x = predicted[it].roundToInt()
            val y = target[it].roundToInt()
            check(x in 0..1 && y in 0..1)

            if (x == y) {
                if (x == 1) a += 1
            } else {
                b += 1
            }
        }

        if (a == n) {
            return 1.0
        }
        if (a == 0) {
            return 0.0
        }

        return 2.0 * a / (2.0 * a + b)
    }
}
