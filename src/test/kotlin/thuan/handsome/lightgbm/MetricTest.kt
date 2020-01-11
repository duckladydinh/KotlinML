package thuan.handsome.lightgbm

import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import thuan.handsome.core.metrics.f1score
import thuan.handsome.core.utils.*
import thuan.handsome.core.utils.getTestData
import thuan.handsome.utils.getTestXSpace

class MetricTest {
    companion object {
        fun testMetricRelevant(
            metric: (DoubleArray, DoubleArray) -> Double,
            kappa: Double,
            enableLog: Boolean = true
        ): Double {
            // val dataPrefix = "data/imblearn_abalone" // good
            val dataPrefix = "data/imblearn_yeast_me2" // not bad, 0.4
            // val dataPrefix = "data/imblearn_mammography" // good
            // val dataPrefix = "data/imblearn_wine_quality" // not bad
            // val dataPrefix = "data/imblearn_letter_img"

            // val dataPrefix = "data/imblearn_abalone_19"
            // val dataPrefix = "data/imblearn_pen_digits"
            // val dataPrefix = "data/imblearn_car_eval_34" // not following rule
            // val dataPrefix = "data/gecco2018_water"

            val (trainData, trainLabel) = getTestData(
                dataPrefix,
                isTest = false
            )
            val (testData, testLabel) = getTestData(
                dataPrefix,
                isTest = true
            )

            val xSpace = getTestXSpace()

            val n = 50
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

                val trainScore = metric.invoke(trainedPreds, trainLabel)
                val testScore = metric.invoke(testPreds, testLabel)

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
    fun testMetricF1Score() {
        val corr = testMetricRelevant(::f1score, -0.25)
        println(corr)
    }

    @Test
    @Disabled
    fun testMetricF1ScoreCorrelation() {
        for (kappa in sequenceOf(
            -2.5, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25,
            0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5
        )) {
            var tot = 0.0
            for (i in 1..5) {
                val corr =
                    testMetricRelevant(::f1score, kappa, enableLog = false)
                tot += corr
            }
            println("($kappa, ${tot / 5})")
        }
    }
}
