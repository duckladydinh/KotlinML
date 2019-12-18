package thuan.handsome.lightgbm

import koma.internal.default.generated.matrix.DefaultDoubleMatrix
import koma.matrix.Matrix
import thuan.handsome.lightgbm.model.Booster
import thuan.handsome.lightgbm.model.Dataset

fun train(params: Map<String, Any>, data: Matrix<Double>, label: IntArray, rounds: Int): Booster {
	val dataset = Dataset.from(data, label)
	return Booster.fit(params, dataset, rounds)
}

fun slice(data: Matrix<Double>, rowIndexes: Collection<Int>): Matrix<Double> {
	val mat = DefaultDoubleMatrix(rows = rowIndexes.size, cols = data.numCols())
	for ((row, dataRow) in rowIndexes.withIndex()) {
		mat.setRow(row, data.getRow(dataRow))
	}
	return mat
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
		val trainData = slice(data, trainSet)
		val trainLabel = label.sliceArray(trainSet)
		val booster = train(params, trainData, trainLabel, rounds)

		val validData = slice(data, validSet)
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