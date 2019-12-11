package thuan.handsome.autoparams

import krangl.*
import org.apache.commons.csv.CSVFormat
import thuan.handsome.autoparams.lightgbm.Booster
import thuan.handsome.autoparams.lightgbm.Dataset
import kotlin.math.min
import kotlin.test.assertEquals

fun prepareFolds(n: Int, nFolds: Int): Pair<List<IntArray>, List<IntArray>> {
	val indexes = IntRange(1, n).shuffled()
	val foldSize = n / nFolds

	val validSets = ArrayList<IntArray>(nFolds)
	for (i in 0 until n step foldSize) {
		val validSet = indexes.subList(i, min(i + foldSize, n))
		validSets.add(validSet.toIntArray())
	}

	val trainSets = ArrayList<IntArray>(nFolds)
	for ((i, validSet) in validSets.withIndex()) {
		val trainSet = ArrayList<Int>(validSet.size)
		for ((j, otherValidSet) in validSets.withIndex()) {
			if (i != j) {
				trainSet.addAll(otherValidSet.asSequence())
			}
		}
		trainSets.add(trainSet.toIntArray())
	}

	return Pair(trainSets, validSets)
}

fun crossValidate(params: Map<String, Any>, df: DataFrame, labelIndex: Int, rounds: Int, nFolds: Int): DoubleArray {
	val (trainSets, validSets) = prepareFolds(df.nrow, nFolds)
	val scores = ArrayList<Double>(trainSets.size)

	for ((trainSet, validSet) in (trainSets zip validSets)) {
		val trainDF = df.slice(*trainSet)
		val (trainData, trainLabel) = getXY(trainDF, labelIndex)
		val dataset = Dataset.from(trainData, trainLabel)
		val booster = Booster.train(dataset, params, rounds)

		val validDF = df.slice(*validSet)
		val (validData, validLabel) = getXY(validDF, labelIndex)
		val preds = booster.predict(validData)
		scores.add(f1score(preds.toBinaryArray(), validLabel))
	}

	return scores.toDoubleArray()
}

fun DoubleArray.toBinaryArray(): IntArray {
	return IntArray(this.size) {
		if (this[it] >= 0.5) 1 else 0
	}
}

fun f1score(predicted: IntArray, target: IntArray): Double {
	assertEquals(predicted.size, target.size)
	val right = intArrayOf(0, 0)
	val wrong = intArrayOf(0, 0)
	val n = predicted.size

	repeat(n) {
		if (target[it] == predicted[it]) {
			right[predicted[it]] += 1
		} else {
			wrong[predicted[it]] += 1
		}
	}

	if (right[0] == n) {
		return 1.0
	}
	if (right[1] == 0) {
		return 0.0
	}

	val precision = right[1].toDouble() / (right[1] + wrong[1])
	val recall = right[1].toDouble() / (right[1] + wrong[0])
	return 2 * (precision * recall) / (precision + recall)
}

fun main() {
	val df = DataFrame.readCSV("data/gecco2018_water_train.csv", format = CSVFormat.DEFAULT.withNullString(""))
	df.print(maxRows = 10)
	val cv = crossValidate(mapOf("objective" to "binary"), df, 0, 100, 5)
	print(cv.joinToString(" "))


	// val (data, label) = getXY(df, 0)
	//
	// val dataset = Dataset.from(data, label)
	// val booster = Booster.train(dataset, mapOf(
	// 	"objective" to "binary"
	// ), 300)
	// booster.save("model.txt")
	//
	// val preds = booster.predict(data)
	// println(preds.toBinaryArray().sum())
	// println(f1score(preds.toBinaryArray(), label))
}

private fun getXY(df: DataFrame, labelIndex: Int): Pair<Array<DoubleArray>, IntArray> {
	val data = Array(df.nrow) {
		DoubleArray(df.ncol - 1) {
			Double.NaN
		}
	}
	val label = IntArray(df.nrow)

	for ((rowIndex, row) in df.rows.withIndex()) {
		for ((index, value) in row.values.withIndex()) {
			if (index == labelIndex) {
				label[rowIndex] = value as Int
			} else if (value != null) {
				data[rowIndex][index - 1] = value as Double
			}
		}
	}

	return Pair(data, label)
}

