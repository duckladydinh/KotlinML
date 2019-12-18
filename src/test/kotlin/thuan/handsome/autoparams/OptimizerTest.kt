package thuan.handsome.autoparams

import org.junit.Test
import thuan.handsome.lightgbm.cv
import thuan.handsome.lightgbm.metric.f1score
import thuan.handsome.lightgbm.toBinaryArray
import thuan.handsome.lightgbm.train
import thuan.handsome.utils.LOGGER
import thuan.handsome.utils.getXY

class OptimizerTest {
	@Test
	fun randomSearch() {
		val (trainData, trainLabel) = getXY("/data/gecco2018_water_train.csv", 0)

		val (_parameters, _) = maximize(
			fun(_params: Map<String, Any>): Double {
				val params = _params + mapOf(
					"objective" to "binary",
					"verbose" to -1,
					"is_unbalance" to false
				)

				val scores = cv(params, trainData, trainLabel, 30, 5, ::f1score)
				return scores.min()!!
			}, listOf(
				Param("feature_fraction", 0.0, 1.0),
				Param("bagging_fraction", 0.0, 1.0),
				Param("max_depth", 10, 30, true),
				Param("min_split_gain", 0.0, 1.0),
				Param("min_child_weight", 1, 10, true)
			), 30
		)
		val parameters = _parameters + mapOf(
			"objective" to "binary",
			"verbose" to -1,
			"is_unbalance" to false
		)

		val booster = train(parameters, trainData, trainLabel, 30)
		val trainedPreds = booster.predict(trainData).toBinaryArray()

		LOGGER.info { "Train F1 = ${f1score(trainedPreds, trainLabel)}" }

		val (testData, testLabel) = getXY("/data/gecco2018_water_test.csv", 0)

		val testPreds = booster.predict(testData).toBinaryArray()
		LOGGER.info { "Test F1 = ${f1score(testPreds, testLabel)}" }

		booster.close()
	}

}