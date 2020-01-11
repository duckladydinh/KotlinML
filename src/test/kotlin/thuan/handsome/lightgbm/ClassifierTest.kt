package thuan.handsome.lightgbm

import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import thuan.handsome.core.metrics.f1score
import thuan.handsome.core.utils.LOGGER
import thuan.handsome.core.utils.getTestData

class ClassifierTest {
    companion object {
        private fun simpleBinaryClassifier(dataPrefix: String) {
            val params = mapOf(
                "objective" to "binary",
                "is_unbalance" to false,
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

            var start = System.currentTimeMillis()
            val scores = Booster.cv(::f1score, params, trainData, trainLabel, 100, 5)
            LOGGER.atInfo().log("CV = ${scores.joinToString(" ")}")
            LOGGER.atInfo().log("CV Time: ${System.currentTimeMillis() - start}")

            start = System.currentTimeMillis()
            val booster = Booster.fit(params, trainData, trainLabel, 100)
            val trainedPreds = booster.predict(trainData)
            LOGGER.atInfo().log("Train Time: ${System.currentTimeMillis() - start}")

            start = System.currentTimeMillis()
            LOGGER.atInfo().log(
                "Train F1 = ${f1score(
                    trainedPreds,
                    trainLabel
                )}"
            )
            LOGGER.atInfo().log("Train Prediction Time: ${System.currentTimeMillis() - start}")

            start = System.currentTimeMillis()
            val testPreds = booster.predict(testData)
            LOGGER.atInfo().log(
                "Test F1 = ${f1score(
                    testPreds,
                    testLabel
                )}"
            )
            LOGGER.atInfo().log("Test Prediction Time: ${System.currentTimeMillis() - start}")

            booster.close()
        }
    }

    @ParameterizedTest(name = "Set #{index} [{arguments}]")
    @ValueSource(
        strings = [
            "data/gecco2018_water",
            "data/imblearn_abalone",
            "data/imblearn_abalone_19",
            "data/imblearn_car_eval_34",
            "data/imblearn_letter_img",
            "data/imblearn_mammography",
            "data/imblearn_pen_digits",
            "data/imblearn_wine_quality",
            "data/imblearn_yeast_me2"
        ]
    )
    fun testDefaultPerformance(dataPrefix: String) {
        simpleBinaryClassifier(dataPrefix)
    }
}
