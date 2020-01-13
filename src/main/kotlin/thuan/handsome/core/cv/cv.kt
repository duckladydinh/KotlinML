package thuan.handsome.core.cv

import koma.matrix.Matrix
import thuan.handsome.core.metrics.Metric
import thuan.handsome.core.predictor.Predictor
import thuan.handsome.core.utils.sliceByRows

fun crossValidate(
    learner: (Matrix<Double>, DoubleArray) -> Predictor,
    metric: Metric,
    data: Matrix<Double>,
    label: DoubleArray,
    nFolds: Int
): DoubleArray {
    val folds = generateFolds(data.numRows(), nFolds)
    return DoubleArray(folds.size) {
        val (trainIndexes, validIndexes) = folds[it]
        val trainData = sliceByRows(data, trainIndexes)
        val trainLabel = label.sliceArray(trainIndexes)
        val validData = sliceByRows(data, validIndexes)
        val validLabel = label.sliceArray(validIndexes)

        val predictor = learner.invoke(trainData, trainLabel)
        val preds = predictor.predict(validData)
        if (predictor is AutoCloseable) {
            predictor.close()
        }
        metric.evaluate(preds, validLabel)
    }
}

fun generateFolds(n: Int, nFolds: Int): List<Pair<List<Int>, List<Int>>> {
    require(n >= nFolds)
    require(nFolds >= 2)

    val validIndexSets = (0 until n).shuffled().withIndex()
        .groupBy {
            it.index % nFolds
        }
        .map {
            it.value.map(IndexedValue<Int>::value)
        }

    val trainIndexSets = ArrayList<List<Int>>(nFolds)
    for (i in validIndexSets.indices) {
        val trainSet = validIndexSets.slice(validIndexSets.indices - i).flatten()
        trainIndexSets.add(trainSet)
    }

    return trainIndexSets zip validIndexSets
}
