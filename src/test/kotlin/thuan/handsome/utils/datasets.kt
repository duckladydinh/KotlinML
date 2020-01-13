package thuan.handsome.utils

import koma.matrix.Matrix
import thuan.handsome.core.utils.getXY

val names = arrayOf(
    "data/imblearn_abalone",
    "data/imblearn_wine_quality",
    "data/imblearn_yeast_me2",
    "data/pima_indians_diabetes",
    "data/nba_logreg"
)

private val datasets = mutableMapOf<String, Pair<Matrix<Double>, DoubleArray>>()

fun getTestData(name: String, isTest: Boolean = false): Pair<Matrix<Double>, DoubleArray> {
    val suffix = if (isTest) "test" else "train"
    val path = (if (name.startsWith("/")) "" else "/") + "${name}_$suffix.csv"

    if (!datasets.containsKey(path)) {
        val data = getXY(
            path, 0
        )
        datasets[path] = data
    }

    return datasets[path]!!
}

@Suppress("unused")
fun getTestData(index: Int, isTest: Boolean = false): Pair<Matrix<Double>, DoubleArray> {
    require(index >= 0 && index < names.size)
    val name = names[index]
    return getTestData(name, isTest)
}
