package thuan.handsome.lightgbm

import krangl.DataFrame
import krangl.readCSV
import org.apache.commons.csv.CSVFormat
import org.junit.Test
import thuan.handsome.autoparams.Param
import thuan.handsome.autoparams.maximize
import thuan.handsome.lightgbm.metric.f1score
import kotlin.test.assertTrue

class BinaryClassificationTest {
	private fun String.toExternalPath(): String {
		assertTrue(this.startsWith("/"), "This method only works with resource path (those starting with '/')!")
		return BinaryClassificationTest::class.java.getResource(this).path
	}

	@Test
	fun default() {
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
		val trainDF = DataFrame.readCSV("/data/gecco2018_water_train.csv".toExternalPath(), format = CSVFormat.DEFAULT.withNullString(""))
		val (trainData, trainLabel) = getXY(trainDF, 0)
		println("IO Time: ${System.currentTimeMillis() - start}")
		start = System.currentTimeMillis()

		val scores = cv(params, trainData, trainLabel, 100, 5, ::f1score)
		println("CV = ${scores.joinToString(" ")}")
		println("CV Time: ${System.currentTimeMillis() - start}")
		start = System.currentTimeMillis()

		val booster = train(params, trainData, trainLabel, 100)
		val trainedPreds = booster.predict(trainData).toBinaryArray()
		println("Train F1 = ${f1score(trainedPreds, trainLabel)}")
		println("Train Time: ${System.currentTimeMillis() - start}")
		start = System.currentTimeMillis()

		val testDF = DataFrame.readCSV("/data/gecco2018_water_test.csv".toExternalPath(), format = CSVFormat.DEFAULT.withNullString(""))
		val (testData, testLabel) = getXY(testDF, 0)
		println("IO Test Time: ${System.currentTimeMillis() - start}")
		start = System.currentTimeMillis()

		val testPreds = booster.predict(testData).toBinaryArray()
		println("Test F1 = ${f1score(testPreds, testLabel)}")
		println("Test Time: ${System.currentTimeMillis() - start}")

		booster.close()
	}

	@Test
	fun objectiveFunction() {
		val trainDF = DataFrame.readCSV("/data/gecco2018_water_train.csv".toExternalPath(), format = CSVFormat.DEFAULT.withNullString(""))
		val (trainData, trainLabel) = getXY(trainDF, 0)

		val (parameters, _) = maximize(fun(params: MutableMap<String, Any>): Double {
			params.putAll(mapOf(
				"objective" to "binary",
				"verbose" to -1,
				"is_unbalance" to false
			))
			val scores = cv(params, trainData, trainLabel, 50, 5, ::f1score)
			return scores.min()!!
		}, listOf(
			Param("feature_fraction", 0.0, 1.0),
			Param("bagging_fraction", 0.0, 1.0),
			Param("max_depth", 10, 30, true),
			Param("min_split_gain", 0.0, 1.0),
			Param("min_child_weight", 1, 10, true)
		), 30)

		parameters.putAll(mapOf(
			"objective" to "binary",
			"verbose" to -1,
			"is_unbalance" to false
		))

		val booster = train(parameters, trainData, trainLabel, 100)
		val trainedPreds = booster.predict(trainData).toBinaryArray()

		println("Train F1 = ${f1score(trainedPreds, trainLabel)}")

		val testDF = DataFrame.readCSV("/data/gecco2018_water_test.csv".toExternalPath(), format = CSVFormat.DEFAULT.withNullString(""))
		val (testData, testLabel) = getXY(testDF, 0)

		val testPreds = booster.predict(testData).toBinaryArray()
		println("Test F1 = ${f1score(testPreds, testLabel)}")

		booster.close()
	}
}

