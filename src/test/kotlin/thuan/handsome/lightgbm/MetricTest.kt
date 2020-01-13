package thuan.handsome.lightgbm

import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import thuan.handsome.TestSettings.Companion.getTestDataPrefix
import thuan.handsome.TestSettings.Companion.getTestMetric
import thuan.handsome.TestSettings.Companion.getTestXSpace
import thuan.handsome.core.metrics.Metric
import thuan.handsome.core.utils.correlationOf
import thuan.handsome.core.utils.getTestData
import thuan.handsome.core.utils.mean
import thuan.handsome.core.utils.std

class MetricTest {
    companion object {
        fun testMetricRelevant(
            metric: Metric,
            kappa: Double,
            enableLog: Boolean = true
        ): Double {
            val xSpace = getTestXSpace()
            val dataPrefix = getTestDataPrefix() // best
            val (trainData, trainLabel) = getTestData(dataPrefix, isTest = false)
            val (testData, testLabel) = getTestData(dataPrefix, isTest = true)

            val n = 100
            val xs = DoubleArray(n)
            val ys = DoubleArray(n)

            for (i in 0 until n) {
                val params = xSpace.decorate(xSpace.sample())
                val scores = Booster.cv(metric, params, trainData, trainLabel, 30, 5)
                val mean = scores.mean()
                val std = scores.std(mean)

                val objScore = mean + kappa * std

                val booster = Booster.fit(params, trainData, trainLabel, 30)
                val trainedPreds = booster.predict(trainData)
                val testPreds = booster.predict(testData)

                val trainScore = metric.evaluate(trainedPreds, trainLabel)
                val testScore = metric.evaluate(testPreds, testLabel)

                xs[i] = objScore
                ys[i] = testScore

                if (enableLog) {
                    println("($objScore, $trainScore, $testScore),")
                }

                booster.close()
            }
            return correlationOf(xs, ys)
        }
    }

    @Test
    @Disabled
    fun testSingleMetricScoreCorrelation() {
        val corr = testMetricRelevant(getTestMetric(), 0.0)
        println(corr)
    }

    @Test
    @Disabled
    fun testMultiMetricScoreCorrelation() {
        val metric = getTestMetric()
        for (kappa in sequenceOf(
            // -2.5, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25,
            0.0
            // , 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5
        )) {
            var tot = 0.0
            for (i in 1..5) {
                val corr =
                    testMetricRelevant(metric, kappa, enableLog = false)
                tot += corr
            }
            println("($kappa, ${tot / 5})")
        }
    }
}
