package thuan.handsome.core.metrics

import kotlin.math.roundToInt

class AccuracyScore : Metric {
    override fun evaluate(predicted: DoubleArray, target: DoubleArray): Double {
        require(predicted.size == target.size)
        require(predicted.isNotEmpty())

        val n = predicted.size
        var corrects = 0.0

        repeat(n) {
            val x = predicted[it].roundToInt()
            val y = target[it].roundToInt()

            if (x == y) {
                corrects += 1
            }
        }

        return corrects / n
    }
}
