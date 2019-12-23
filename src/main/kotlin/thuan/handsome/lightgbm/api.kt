package thuan.handsome.lightgbm

import koma.matrix.Matrix
import thuan.handsome.lightgbm.model.Booster
import thuan.handsome.lightgbm.model.Dataset
import thuan.handsome.utils.sliceByRows

fun train(params: Map<String, Any>, data: Matrix<Double>, label: IntArray, rounds: Int): Booster {
	val dataset = Dataset.from(data, label)
	return Booster.fit(params, dataset, rounds)
}

fun cv(
	params: Map<String, Any>,
	data: Matrix<Double>,
	label: IntArray,
	rounds: Int,
	nFolds: Int,
	metric: (IntArray, IntArray) -> Double
): DoubleArray {
	val folds = makeFolds(data.numRows(), nFolds)
	val scores = ArrayList<Double>(folds.size)

	for ((trainSet, validSet) in folds) {
		val trainData = sliceByRows(data, trainSet)
		val trainLabel = label.sliceArray(trainSet)
		val booster = train(params, trainData, trainLabel, rounds)

		val validData = sliceByRows(data, validSet)
		val validLabel = label.sliceArray(validSet)
		val preds = booster.predict(validData)
		booster.close()

		scores.add(metric.invoke(preds, validLabel))
	}

	return scores.toDoubleArray()
}

private fun makeFolds(n: Int, nFolds: Int): List<Pair<List<Int>, List<Int>>> {
	val validSets = IntRange(0, n - 1).shuffled().withIndex()
		.groupBy { it.index % nFolds }
		.map { indexed ->
			indexed.value.map {
				it.value
			}
		}

	val trainSets = ArrayList<List<Int>>(nFolds)
	for (i in validSets.indices) {
		val trainSet = validSets.slice(IntRange(0, validSets.size - 1) - i).flatten()
		trainSets.add(trainSet)
	}

	return trainSets zip validSets
}