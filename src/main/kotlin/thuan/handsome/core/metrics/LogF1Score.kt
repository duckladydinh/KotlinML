package thuan.handsome.core.metrics

import kotlin.math.ln

class LogF1Score(private val eps: Double = 1e-9) : Metric {
    private val f1Score = F1Score()

    override fun evaluate(predicted: DoubleArray, target: DoubleArray): Double {
        return ln(f1Score.evaluate(predicted, target) + eps)
    }
}
