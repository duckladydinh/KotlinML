package thuan.handsome.core.utils

import koma.matrix.Matrix

val names = arrayOf(
    "gecco2018_water",
    "imblearn_abalone",
    "imblearn_abalone_19",
    "imblearn_car_eval_34",
    "imblearn_letter_img",
    "imblearn_mammography",
    "imblearn_pen_digits",
    "imblearn_wine_quality",
    "imblearn_yeast_me2"
)

private val datasets = mutableMapOf<String, Pair<Matrix<Double>, DoubleArray>>()

fun getTestData(name: String, isTest: Boolean = false): Pair<Matrix<Double>, DoubleArray> {
    val suffix = if (isTest) "test" else "train"
    val path = "${name}_$suffix.csv"

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
