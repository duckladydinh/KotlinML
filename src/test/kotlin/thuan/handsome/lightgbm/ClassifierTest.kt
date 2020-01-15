package thuan.handsome.lightgbm

import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import thuan.handsome.TestSettings.Companion.getTestData
import thuan.handsome.TestSettings.Companion.getTestMetric
import thuan.handsome.core.metrics.Metric

class ClassifierTest {
    companion object {
        private fun simpleBinaryClassifier(dataPrefix: String, metric: Metric): Pair<Double, Double> {
            val params = mapOf(
                "objective" to "binary",
                "is_unbalance" to true,

                // "min_child_weight" to 16,
                // "num_leaves" to 96,
                // "max_depth" to 10,
                // "min_split_gain" to 0.01,
                // "subsample" to 0.8,
                // "lambda_l1" to 0.9,
                // "lambda_l2" to 0.9,

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
    fun testDefaultPerformance(dataPrefix: String) {
        val metric = getTestMetric()
        var aTot = 0.0
        var bTot = 0.0
        for (i in 1..5) {
            val (a, b) = simpleBinaryClassifier(dataPrefix, metric)
            aTot += a
            bTot += b
        }
        println("(Mean Train, Mean Test),")
        println("(${aTot / 5}, ${bTot / 5}),")
    }
}
