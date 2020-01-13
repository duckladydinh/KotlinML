@file:Suppress("unused")

package thuan.handsome.core.metrics

interface Metric {
    fun evaluate(predicted: DoubleArray, target: DoubleArray): Double
}
