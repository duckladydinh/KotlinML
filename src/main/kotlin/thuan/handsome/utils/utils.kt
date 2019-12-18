package thuan.handsome.utils

import krangl.DataFrame
import krangl.readCSV
import mu.KotlinLogging
import org.apache.commons.csv.CSVFormat

private fun getXY(df: DataFrame, labelIndex: Int): Pair<Array<DoubleArray>, IntArray> {
	val data = Array(df.nrow) {
		DoubleArray(df.ncol - 1) { Double.NaN }
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

class PathIO {
	companion object {
		fun getExternalPathForResource(resourcePath: String): String {
			return PathIO::class.java.getResource(resourcePath).path
		}
	}
}

val LOGGER = KotlinLogging.logger {}

fun getXY(csvMatrixPath: String, labelColumnIndex: Int): Pair<Array<DoubleArray>, IntArray> {
	val df = DataFrame.readCSV(
		if (csvMatrixPath.startsWith("/")) PathIO.getExternalPathForResource(csvMatrixPath) else csvMatrixPath,
		format = CSVFormat.DEFAULT.withNullString("")
	)

	return getXY(df, labelColumnIndex)
}
