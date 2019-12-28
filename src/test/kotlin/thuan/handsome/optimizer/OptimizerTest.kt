package thuan.handsome.optimizer

import kotlin.math.pow
import kotlin.math.sqrt
import org.junit.Test
import thuan.handsome.core.metrics.f1score
import thuan.handsome.core.utils.LOGGER
import thuan.handsome.core.utils.getXY
import thuan.handsome.core.xspace.UniformXSpace
import thuan.handsome.core.xspace.XType
import thuan.handsome.lightgbm.Booster

class OptimizerTest {
    private companion object {
        fun ucb(x: DoubleArray, kappa: Double = 2.576): Double {
            val mean = x.sum() / x.size
            val std = sqrt(
                // variance
                x.map { (it - mean).pow(2) }.sum() / (x.size - 1)
            )
            return mean + kappa * std
        }

        fun testOptimizer(optimizer: Optimizer) {
            val (trainData, trainLabel) = getXY(
                "/data/gecco2018_water_train.csv", 0
            )
            val (testData, testLabel) = getXY(
                "/data/gecco2018_water_test.csv", 0
            )

            val xSpace = UniformXSpace().apply {
                addConstantParams(
                    mapOf(
                        "objective" to "binary",
                        "is_unbalance" to false,
                        "verbose" to -1
                    )
                )
                addParam("feature_fraction", 0.1, 0.9)
                addParam("bagging_fraction", 0.8, 1.0)
                addParam("max_depth", 17.0, 25.0, XType.INT)
                addParam("num_leaves", 50.0, 500.0, XType.INT)
                addParam("min_split_gain", 0.001, 0.1)
                addParam("min_child_weight", 10.0, 25.0, XType.INT)
            }

            val (params, _) = optimizer.argmax(
                fun(params: Map<String, Any>): Double {
                    val scores = Booster.cv(::f1score, params, trainData, trainLabel, 30, 5)
                    return ucb(scores)
                },
                xSpace,
                20
            )

            val booster = Booster.fit(params, trainData, trainLabel, 30)
            val trainedPreds = booster.predict(trainData)

            LOGGER.info {
                "Train F1 = ${f1score(trainedPreds, trainLabel)}"
            }

            val testPreds = booster.predict(testData)
            LOGGER.info {
                "Test F1 = ${f1score(testPreds, testLabel)}"
            }

            booster.close()
        }
    }

    @Test
    fun randomOptimizer() {
        testOptimizer(UniformOptimizer())
    }

    @Test
    fun bayesianOptimizer() {
        testOptimizer(BayesianOptimizer())
    }
}
