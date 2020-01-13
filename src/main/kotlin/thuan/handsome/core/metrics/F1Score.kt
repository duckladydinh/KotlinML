package thuan.handsome.core.metrics

import kotlin.math.roundToInt

class F1Score : Metric {
    override fun evaluate(predicted: DoubleArray, target: DoubleArray): Double {
        require(predicted.size == target.size)
        require(predicted.isNotEmpty())

        // TN, TP, FN, FP
        val stats = intArrayOf(0, 0, 0, 0)
        val n = predicted.size

        repeat(n) {
            val x = predicted[it].roundToInt()
            val y = target[it].roundToInt()
            check(x in 0..1 && y in 0..1)

            if (x == y) {
                stats[x] += 1
            } else {
                stats[x + 2] += 1
            }
        }

        if (stats[0] == n) {
            return 1.0
        }
        if (stats[1] == 0) {
            return 0.0
        }

        val precision = stats[1].toDouble() / (stats[1] + stats[3])
        val recall = stats[1].toDouble() / (stats[1] + stats[2])

        return 2 * (precision * recall) / (precision + recall)
    }
}
