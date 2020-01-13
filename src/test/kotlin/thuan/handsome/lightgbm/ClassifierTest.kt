package thuan.handsome.lightgbm

import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import thuan.handsome.core.metrics.f1score
import thuan.handsome.core.utils.getTestData

class ClassifierTest {
    companion object {
        private fun simpleBinaryClassifier(dataPrefix: String): Pair<Double, Double> {
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

            val scores = Booster.cv(::f1score, params, trainData, trainLabel, 100, 5)

            val booster = Booster.fit(params, trainData, trainLabel, 100)
            val trainedPreds = booster.predict(trainData)

            val trainScore = f1score(
                trainedPreds,
                trainLabel
            )

            val testPreds = booster.predict(testData)
            val testScore = f1score(
                testPreds,
                testLabel
            )
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
        var aTot = 0.0
        var bTot = 0.0
        for (i in 1..10) {
            val (a, b) = simpleBinaryClassifier(dataPrefix)
            aTot += a
            bTot += b
        }
        println("(Mean Train, Mean Test),")
        println("(${aTot / 10}, ${bTot / 10}),")
    }
}
