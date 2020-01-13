package thuan.handsome.optimizer

import org.junit.jupiter.api.Test
import thuan.handsome.core.metrics.f1score
import thuan.handsome.core.utils.getTestData
import thuan.handsome.core.utils.mean
import thuan.handsome.lightgbm.Booster
import thuan.handsome.utils.getTestXSpace

/**
 * This test has no validation since I only want to see how it runs,
 * and ensure that there is no runtime exception
 */
class OptimizerTest {
    private companion object {
        fun testOptimizer(optimizer: Optimizer, metric: (DoubleArray, DoubleArray) -> Double): Pair<Double, Double> {
            // val dataPrefix = "data/imblearn_abalone" // good
            // val dataPrefix = "data/imblearn_wine_quality" // not bad
            // val dataPrefix = "data/imblearn_yeast_me2" // not bad, 0.4
            // val dataPrefix = "data/pima_indians_diabetes" // good
            val dataPrefix = "data/nba_logreg" // best

            val (trainData, trainLabel) = getTestData(
                dataPrefix,
                isTest = false
            )
            val (testData, testLabel) = getTestData(
                dataPrefix,
                isTest = true
            )
            val xSpace = getTestXSpace()

            val (params, _) = optimizer.argmax(
                fun(params: Map<String, Any>): Double {
                    val scores = Booster.cv(metric, params, trainData, trainLabel, 30, 5)
                    return scores.mean()
                },
                xSpace,
                16
            )

            val booster = Booster.fit(params, trainData, trainLabel, 30)
            val trainedPreds = booster.predict(trainData)

            val trainScore = metric.invoke(trainedPreds, trainLabel)
            val testPreds = booster.predict(testData)
            val testScore = metric.invoke(testPreds, testLabel)

            println("Train F1 = $trainScore | Test F1 = $testScore")

            booster.close()

            return Pair(trainScore, testScore)
        }

        fun multiTest(optimizer: Optimizer) {
            var totTrain = 0.0
            var totTest = 0.0
            val n = 10

            val startTime = System.currentTimeMillis()

            for (i in 1..n) {
                val (train, test) = testOptimizer(optimizer, ::f1score)
                totTrain += train
                totTest += test
            }

            println("Mean Train: ${totTrain / n} | Mean Test: ${totTest / n} | Duration: ${System.currentTimeMillis() - startTime}")
        }
    }

    @Test
    fun randomOptimizerWithF1Score() {
        multiTest(UniformOptimizer())
    }

    @Test
    fun bayesianOptimizerWithF1Score() {
        multiTest(BayesianOptimizer())
    }
}
