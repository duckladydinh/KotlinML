package thuan.handsome.optimizer

import org.junit.jupiter.api.Test
import thuan.handsome.TestSettings.Companion.getTestData
import thuan.handsome.TestSettings.Companion.getTestDataPrefix
import thuan.handsome.TestSettings.Companion.getTestMetric
import thuan.handsome.TestSettings.Companion.getTestXSpace
import thuan.handsome.core.metrics.F1Score
import thuan.handsome.core.metrics.Metric
import thuan.handsome.core.optimizer.BayesianOptimizer
import thuan.handsome.core.optimizer.Optimizer
import thuan.handsome.core.optimizer.RandomOptimizer
import thuan.handsome.core.utils.mean
import thuan.handsome.lightgbm.Booster

/**
 * This test has no validation since I only want to see how it runs,
 * and ensure that there is no runtime exception
 */
class OptimizerTest {
    private companion object {
        val f1Score = F1Score()

        fun testOptimizer(optimizer: Optimizer, metric: Metric): Pair<Double, Double> {
            val xSpace = getTestXSpace()
            val dataPrefix = getTestDataPrefix()
            val (trainData, trainLabel) = getTestData(dataPrefix, isTest = false)
            val (testData, testLabel) = getTestData(dataPrefix, isTest = true)

            val func = fun(params: Map<String, Any>): Double {
                val scores = Booster.cv(metric, params,
                    data = trainData,
                    label = trainLabel,
                    maxiter = 30,
                    nFolds = 5
                )
                return scores.mean()
            }
            val (params, _) = optimizer.argmax(
                func, xSpace,
                maxiter = 32
            )

            val booster = Booster.fit(params, trainData, trainLabel, 30)
            val trainedPreds = booster.predict(trainData)

            val trainScore = f1Score.evaluate(trainedPreds, trainLabel)
            val testPreds = booster.predict(testData)
            val testScore = f1Score.evaluate(testPreds, testLabel)

            println("Train F1 = $trainScore | Test F1 = $testScore")

            booster.close()

            return Pair(trainScore, testScore)
        }

        fun multiTest(optimizer: Optimizer, n: Int = 8) {
            val metric = getTestMetric()

            var totTrain = 0.0
            var totTest = 0.0

            val startTime = System.currentTimeMillis()

            for (i in 1..n) {
                val (train, test) = testOptimizer(optimizer, metric)
                totTrain += train
                totTest += test
            }

            println("Mean Train: ${totTrain / n} | Mean Test: ${totTest / n} | Duration: ${System.currentTimeMillis() - startTime}")
        }
    }

    @Test
    fun randomOptimizerTest() {
        multiTest(RandomOptimizer())
    }

    @Test
    fun bayesianOptimizerTest() {
        multiTest(BayesianOptimizer())
    }
}
