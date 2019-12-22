package thuan.handsome.autoparams

import org.junit.Test
import thuan.handsome.autoparams.optimizer.UniformRandomOptimizer
import thuan.handsome.autoparams.xspace.UniformXSpace
import thuan.handsome.lightgbm.cv
import thuan.handsome.ml.f1score
import thuan.handsome.lightgbm.train
import thuan.handsome.ml.LOGGER
import thuan.handsome.ml.getXY

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
				"is_unbalance" to false
			)
		)
		xSpace.addParam("feature_fraction", 0.0, 1.0)
		xSpace.addParam("bagging_fraction", 0.0, 1.0)
		xSpace.addParam("max_depth", 10.0, 30.0, false)
		xSpace.addParam("min_split_gain", 0.0, 1.0)
		xSpace.addParam("min_child_weight", 1.0, 10.0, false)

		val optimizer = UniformRandomOptimizer()

		val (params, _) = optimizer.argMaximize(
			fun(params: Map<String, Any>): Double {
				val scores = cv(params, trainData, trainLabel, 30, 5, ::f1score)
				return scores.min()!!
			},
			xSpace,
			30
		)

		val booster = train(params, trainData, trainLabel, 30)
		val trainedPreds = booster.predict(trainData)

		LOGGER.info { "Train F1 = ${f1score(
			trainedPreds,
			trainLabel
		)}" }

		val (testData, testLabel) = getXY(
			"/data/gecco2018_water_test.csv",
			0
		)

		val testPreds = booster.predict(testData)
		LOGGER.info { "Test F1 = ${f1score(
			testPreds,
			testLabel
		)}" }

		booster.close()
	}
}