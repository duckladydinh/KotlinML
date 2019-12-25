package thuan.handsome.ml

interface DifferentialFunction {
    companion object {
        fun from(epsilon: Double = 1e-8, func: (DoubleArray) -> Double): DifferentialFunction {
            return object : DifferentialFunction {
                override fun evaluate(x: DoubleArray): DifferentialEvaluation {
                    val y = func.invoke(x)
                    val grads = gradientsOf(func, x, y, epsilon)
                    return DifferentialEvaluation(y, grads)
                }
            }
        }

        private fun gradientsOf(
            func: (DoubleArray) -> Double,
            xZero: DoubleArray,
            yZero: Double,
            epsilon: Double
        ): DoubleArray {
            require(epsilon > 0)

            return DoubleArray(xZero.size) {
                xZero[it] += epsilon
                val y = func.invoke(xZero)
                val gradient = (y - yZero) / epsilon
                xZero[it] -= epsilon

                gradient
            }
        }
    }

    fun evaluate(x: DoubleArray): DifferentialEvaluation
}
