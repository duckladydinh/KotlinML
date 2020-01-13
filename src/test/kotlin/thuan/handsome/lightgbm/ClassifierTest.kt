package thuan.handsome.lightgbm

import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import thuan.handsome.core.metrics.Metric
import thuan.handsome.core.utils.getTestData
import thuan.handsome.utils.getTestMetric

class ClassifierTest {
    companion object {
        private fun simpleBinaryClassifier(dataPrefix: String, metric: Metric): Pair<Double, Double> {
            val params = mapOf(
                "objective" to "binary",
                "is_unbalance" to true,
                "verbose" to -1
            )

            val (trainData, trainLabel) = getTestData(
                dataPrefix,
                isTest = false
            )
            val (testData, testLabel) = getTestData(
                dataPrefix,
                isTest = true
            )

            val booster = Booster.fit(params, trainData, trainLabel, 100)
            val trainedPreds = booster.predict(trainData)
            val trainScore = metric.evaluate(trainedPreds, trainLabel)

            val testPreds = booster.predict(testData)
            val testScore = metric.evaluate(testPreds, testLabel)
            booster.close()

            return Pair(trainScore, testScore)
        }
    }

    @ParameterizedTest(name = "Set #{index} [{arguments}]")
    @ValueSource(
        strings = [
            "data/imblearn_abalone",
            "data/imblearn_wine_quality",
            "data/imblearn_yeast_me2",
            "data/pima_indians_diabetes",
            "data/nba_logreg"
        ]
    )
    fun testDefaultPerformance(dataPrefix: String, metric: Metric = getTestMetric()) {
        var aTot = 0.0
        var bTot = 0.0
        for (i in 1..10) {
            val (a, b) = simpleBinaryClassifier(dataPrefix, metric)
            aTot += a
            bTot += b
        }
        println("(Mean Train, Mean Test),")
        println("(${aTot / 10}, ${bTot / 10}),")
    }
}
