package thuan.handsome.lightgbm

import thuan.handsome.lightgbm.model.Booster
import thuan.handsome.lightgbm.model.Dataset

fun train(params: Map<String, Any>, data: Array<DoubleArray>, label: IntArray, rounds: Int): Booster {
	val dataset = Dataset.from(data, label)
	return Booster.fit(params, dataset, rounds)
}

fun cv(
	params: Map<String, Any>,
	data: Array<DoubleArray>,
	label: IntArray,
	rounds: Int,
	nFolds: Int,
	metric: (IntArray, IntArray) -> Double
): DoubleArray {
	val folds = makeFolds(data.size, nFolds)
	val scores = ArrayList<Double>(folds.size)

	for ((trainSet, validSet) in folds) {
		val trainData = data.sliceArray(trainSet)
		val trainLabel = label.sliceArray(trainSet)
		val booster = train(params, trainData, trainLabel, rounds)

		val validData = data.sliceArray(validSet)
		val validLabel = label.sliceArray(validSet)
		val preds = booster.predict(validData)
		booster.close()

		scores.add(metric.invoke(preds.toBinaryArray(), validLabel))
	}

	return scores.toDoubleArray()
}

fun makeFolds(n: Int, nFolds: Int): List<Pair<List<Int>, List<Int>>> {
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

fun DoubleArray.toBinaryArray(): IntArray {
	return IntArray(this.size) {
		if (this[it] >= 0.5) 1 else 0
	}
}