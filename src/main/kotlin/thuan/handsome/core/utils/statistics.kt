package thuan.handsome.core.utils

import kotlin.math.pow
import kotlin.math.sqrt
import krangl.mean

fun DoubleArray.mean(): Double {
    return this.sum() / this.size
}

fun DoubleArray.variance(mu: Double = this.mean(), sample: Boolean = true): Double {
    return this.map { (it - mu).pow(2) }.sum() / (this.size - if (sample) 1 else 0)
}

fun DoubleArray.std(mu: Double = this.mean(), sample: Boolean = true): Double {
    return sqrt(this.variance(mu, sample))
}

fun correlationOf(x: DoubleArray, y: DoubleArray): Double {
    require(x.size == y.size)

    val xyMean = (x zip y).map { it.first * it.second }.mean()
    val xMean = x.mean()
    val yMean = y.mean()

    return (xyMean - xMean * yMean) / (x.std(xMean, sample = false) * y.std(yMean, sample = false))
}
