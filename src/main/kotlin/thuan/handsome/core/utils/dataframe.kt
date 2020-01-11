package thuan.handsome.core.utils

import koma.extensions.set
import koma.matrix.Matrix
import koma.zeros
import krangl.DataFrame
import krangl.readCSV
import org.apache.commons.csv.CSVFormat

fun getXY(csvMatrixPath: String, labelColumnIndex: Int): Pair<Matrix<Double>, DoubleArray> {
    val df = DataFrame.readCSV(
        csvMatrixPath,
        format = CSVFormat.DEFAULT.withNullString("")
    )

    return getXY(df, labelColumnIndex)
}

fun sliceByRows(data: Matrix<Double>, rowIndexes: Collection<Int>): Matrix<Double> {
    val mat = zeros(rowIndexes.size, data.numCols())
    for ((row, dataRow) in rowIndexes.withIndex()) {
        mat.setRow(row, data.getRow(dataRow))
    }
    return mat
}

private fun getXY(df: DataFrame, labelIndex: Int): Pair<Matrix<Double>, DoubleArray> {
    val data = zeros(df.nrow, df.ncol - 1)
    val label = DoubleArray(df.nrow)

    for ((rowIndex, row) in df.rows.withIndex()) {
        for ((colIndex, value) in row.values.withIndex()) {
            if (colIndex == labelIndex) {
                label[rowIndex] = value.toUncheckedDouble()
            } else {
                data[rowIndex, colIndex - 1] = (value ?: Double.NaN) as Double
            }
        }
    }

    return Pair(data, label)
}
