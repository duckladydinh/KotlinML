package thuan.handsome.optimizer

import org.junit.Test
import thuan.handsome.core.utils.*
import thuan.handsome.core.xspace.UniformXSpace
import thuan.handsome.core.xspace.XType
import thuan.handsome.lightgbm.Booster

class OptimizerTest {
    @Test
    fun randomOptimizer() {
        val (trainData, trainLabel) = getXY(
            "/data/gecco2018_water_train.csv",
            0
        )
        val xSpace = UniformXSpace()
        xSpace.addConstantParams(
            mapOf(
                "objective" to "binary",
                "verbose" to -1,
                "is_unbalance" to true
            )
        )
        xSpace.addParam("feature_fraction", 0.1, 0.9)
        xSpace.addParam("bagging_fraction", 0.8, 1.0)
        xSpace.addParam("max_depth", 17.0, 25.0, XType.INT)
        xSpace.addParam("num_leaves", 50.0, 500.0, XType.INT)
        xSpace.addParam("min_split_gain", 0.001, 0.1)
        xSpace.addParam("min_child_weight", 10.0, 25.0, XType.INT)

        val optimizer = UniformOptimizer()

        val (params, _) = optimizer.argmax(
            fun(params: Map<String, Any>): Double {
                val scores = Booster.cv(params, trainData, trainLabel, 30, 5, ::f1score)
                return scores.min()!!
            },
            xSpace,
            30
        )

        val booster = Booster.fit(params, trainData, trainLabel, 30)
        val trainedPreds = booster.predict(trainData)

        LOGGER.info {
            "Train F1 = ${f1score(
                trainedPreds,
                trainLabel
            )}"
        }

        val (testData, testLabel) = getXY(
            "/data/gecco2018_water_test.csv",
            0
        )

        val testPreds = booster.predict(testData)
        LOGGER.info {
            "Test F1 = ${f1score(
                testPreds,
                testLabel
            )}"
        }

        booster.close()
    }

    @Test
    fun bayesianOptimizer() {
        val (trainData, trainLabel) = getXY(
            "/data/gecco2018_water_train.csv",
            0
        )
        val xSpace = UniformXSpace()
        xSpace.addConstantParams(
            mapOf(
                "objective" to "binary",
                "verbose" to -1,
                "is_unbalance" to true
            )
        )
        xSpace.addParam("feature_fraction", 0.1, 0.9)
        xSpace.addParam("bagging_fraction", 0.8, 1.0)
        xSpace.addParam("max_depth", 17.0, 25.0, XType.INT)
        xSpace.addParam("num_leaves", 50.0, 500.0, XType.INT)
        xSpace.addParam("min_split_gain", 0.001, 0.1)
        xSpace.addParam("min_child_weight", 10.0, 25.0, XType.INT)

        val optimizer = BayesianOptimizer()

        val (params, _) = optimizer.argmax(
            fun(params: Map<String, Any>): Double {
                val scores = Booster.cv(params, trainData, trainLabel, 30, 5, ::f1score)
                return scores.min()!!
            },
            xSpace,
            20
        )

        val booster = Booster.fit(params, trainData, trainLabel, 30)
        val trainedPreds = booster.predict(trainData)

        LOGGER.info {
            "Train F1 = ${f1score(
                trainedPreds,
                trainLabel
            )}"
        }

        val (testData, testLabel) = getXY(
            "/data/gecco2018_water_test.csv",
            0
        )

        val testPreds = booster.predict(testData)
        LOGGER.info {
            "Test F1 = ${f1score(
                testPreds,
                testLabel
            )}"
        }

        booster.close()
    }
}
