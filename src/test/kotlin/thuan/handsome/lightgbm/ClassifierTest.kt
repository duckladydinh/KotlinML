package thuan.handsome.lightgbm

import org.junit.Test
import thuan.handsome.ml.utils.*

class ClassifierTest {
    @Test
    fun simpleBinaryClassifier() {
        val params = mapOf(
            "objective" to "binary",
            "verbose" to -1,
            "num_leaves" to 100,
            "feature_fraction" to 0.8,
            "bagging_fraction" to 0.8,
            "max_depth" to 30,
            "min_split_gain" to 0.5,
            "min_child_weight" to 1,
            "is_unbalance" to false
        )

        var start = System.currentTimeMillis()
        val (trainData, trainLabel) = getXY(
            "/data/gecco2018_water_train.csv",
            0
        )
        LOGGER.info { "IO Time: ${System.currentTimeMillis() - start}" }
        start = System.currentTimeMillis()

        val scores = cv(params, trainData, trainLabel, 100, 5, ::f1score)
        LOGGER.info { "CV = ${scores.joinToString(" ")}" }
        LOGGER.info { "CV Time: ${System.currentTimeMillis() - start}" }
        start = System.currentTimeMillis()

        val booster = train(params, trainData, trainLabel, 100)
        val trainedPreds = booster.predict(trainData)
        LOGGER.info {
            "Train F1 = ${f1score(
                trainedPreds,
                trainLabel
            )}"
        }
        LOGGER.info { "Train Time: ${System.currentTimeMillis() - start}" }
        start = System.currentTimeMillis()

        val (testData, testLabel) = getXY(
            "/data/gecco2018_water_test.csv",
            0
        )
        LOGGER.info { "IO Test Time: ${System.currentTimeMillis() - start}" }
        start = System.currentTimeMillis()

        val testPreds = booster.predict(testData)
        LOGGER.info {
            "Test F1 = ${f1score(
                testPreds,
                testLabel
            )}"
        }
        LOGGER.info { "Test Time: ${System.currentTimeMillis() - start}" }

        booster.close()
    }
}
