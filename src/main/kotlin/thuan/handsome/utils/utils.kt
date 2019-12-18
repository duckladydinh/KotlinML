package thuan.handsome.utils

import koma.internal.default.generated.matrix.DefaultDoubleMatrix
import koma.matrix.Matrix
import krangl.DataFrame
import krangl.readCSV
import mu.KotlinLogging
import org.apache.commons.csv.CSVFormat

private fun getXY(df: DataFrame, labelIndex: Int): Pair<Matrix<Double>, IntArray> {
	val cols = df.ncol - 1
	val data = DefaultDoubleMatrix(df.nrow, cols)
	val label = IntArray(df.nrow)

	for ((rowIndex, row) in df.rows.withIndex()) {
		for ((index, value) in row.values.withIndex()) {
			if (index == labelIndex) {
				label[rowIndex] = value as Int
			} else if (value != null) {
				data.setDouble(rowIndex * cols + index - 1, value as Double)
			}
		}
	}

	return Pair(data, label)
}

class PathIO {
	companion object {
		fun getExternalPathForResource(resourcePath: String): String {
			return PathIO::class.java.getResource(resourcePath).path
		}
	}
}

val LOGGER = KotlinLogging.logger {}

fun getXY(csvMatrixPath: String, labelColumnIndex: Int): Pair<Matrix<Double>, IntArray> {
	val df = DataFrame.readCSV(
		if (csvMatrixPath.startsWith("/")) PathIO.getExternalPathForResource(csvMatrixPath) else csvMatrixPath,
		format = CSVFormat.DEFAULT.withNullString("")
	)

	return getXY(df, labelColumnIndex)
}

fun slice(data: Matrix<Double>, rowIndexes: Collection<Int>): Matrix<Double> {
	val mat = DefaultDoubleMatrix(rows = rowIndexes.size, cols = data.numCols())
	for ((row, dataRow) in rowIndexes.withIndex()) {
		mat.setRow(row, data.getRow(dataRow))
	}
	return mat
}
