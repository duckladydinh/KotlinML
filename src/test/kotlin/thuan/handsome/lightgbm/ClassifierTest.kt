package thuan.handsome.lightgbm

import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import thuan.handsome.core.metrics.f1score
import thuan.handsome.core.utils.LOGGER
import thuan.handsome.utils.getTestData

class ClassifierTest {
    companion object {
        private fun simpleBinaryClassifier(dataPrefix: String) {
            val params = mapOf(
                "objective" to "binary",
                "is_unbalance" to false,
                "verbose" to -1
            )

            val (trainData, trainLabel) = getTestData(dataPrefix, isTest = false)
            val (testData, testLabel) = getTestData(dataPrefix, isTest = true)

            var start = System.currentTimeMillis()
            val scores = Booster.cv(::f1score, params, trainData, trainLabel, 100, 5)
            LOGGER.info { "CV = ${scores.joinToString(" ")}" }
            LOGGER.info { "CV Time: ${System.currentTimeMillis() - start}" }

            start = System.currentTimeMillis()
            val booster = Booster.fit(params, trainData, trainLabel, 100)
            val trainedPreds = booster.predict(trainData)
            LOGGER.info { "Train Time: ${System.currentTimeMillis() - start}" }

            start = System.currentTimeMillis()
            LOGGER.info {
                "Train F1 = ${f1score(
                    trainedPreds,
                    trainLabel
                )}"
            }
            LOGGER.info { "Train Prediction Time: ${System.currentTimeMillis() - start}" }

            start = System.currentTimeMillis()
            val testPreds = booster.predict(testData)
            LOGGER.info {
                "Test F1 = ${f1score(
                    testPreds,
                    testLabel
                )}"
            }
            LOGGER.info { "Test Prediction Time: ${System.currentTimeMillis() - start}" }

            booster.close()
        }
    }

    @ParameterizedTest(name = "Set #{index} [{arguments}]")
    @ValueSource(
        strings = [
            "gecco2018_water",
            "imblearn_abalone",
            "imblearn_abalone_19",
            "imblearn_car_eval_34",
            "imblearn_letter_img",
            "imblearn_mammography",
            "imblearn_pen_digits",
            "imblearn_wine_quality",
            "imblearn_yeast_me2"
        ]
    )
    fun testDefaultPerformance(dataPrefix: String) {
        simpleBinaryClassifier(dataPrefix)
    }
}
