package thuan.handsome

import koma.matrix.Matrix
import thuan.handsome.core.metrics.F1Score
import thuan.handsome.core.metrics.Metric
import thuan.handsome.core.utils.getXY
import thuan.handsome.core.xspace.UniformXSpace
import thuan.handsome.core.xspace.XSpace
import thuan.handsome.core.xspace.XType

class TestSettings {
    companion object {
        private val names = arrayOf(
            "data/imblearn_abalone",
            "data/imblearn_wine_quality",
            "data/imblearn_yeast_me2",
            "data/pima_indians_diabetes",
            "data/nba_logreg"
        )

        private val DATASETS = mutableMapOf<String, Pair<Matrix<Double>, DoubleArray>>()

        fun getTestData(name: String, isTest: Boolean = false): Pair<Matrix<Double>, DoubleArray> {
            val suffix = if (isTest) "test" else "train"
            val path = "${name}_$suffix.csv"

            if (!DATASETS.containsKey(path)) {
                val data = getXY(
                    path, 0
                )
                DATASETS[path] = data
            }

            return DATASETS[path]!!
        }

        @Suppress("unused")
        fun getTestData(index: Int, isTest: Boolean = false): Pair<Matrix<Double>, DoubleArray> {
            require(index >= 0 && index < names.size)
            val name = names[index]
            return getTestData(name, isTest)
        }

        private val metric = F1Score()
        // private val metric = LogF1Score()

        private val xSpace = UniformXSpace().apply {
            addConstantParams(
                mapOf(
                    "objective" to "binary",
                    "is_unbalance" to true,
                    "verbose" to -1
                )
            )

            addParam("min_child_weight", 10.0, 25.0, XType.INT)
            addParam("num_leaves", 63.0, 127.0, XType.INT)
            addParam("max_depth", 10.0, 25.0, XType.INT)
            addParam("min_split_gain", 1e-3, 1e-1)
            addParam("learning_rate", 1e-2, 2e-1)
            addParam("subsample", 0.6, 0.999)
            addParam("lambda_l1", 1e-9, 1.0)
            addParam("lambda_l2", 1e-9, 1.0)
        }

        fun getTestXSpace(): XSpace {
            return xSpace
        }

        fun getTestMetric(): Metric {
            return metric
        }

        fun getTestDataPrefix(): String {
    return "data/imblearn_abalone" // good
            // return "data/imblearn_wine_quality" // not bad
            // return "data/imblearn_yeast_me2" // not bad, 0.4
    // return "data/pima_indians_diabetes" // good
            // return "data/nba_logreg" // good
        }
    }
}
