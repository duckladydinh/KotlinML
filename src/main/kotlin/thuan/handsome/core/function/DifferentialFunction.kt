package thuan.handsome.core.function

class DifferentialFunction(
    private val func: (DoubleArray) -> DifferentialEvaluation
) : (DoubleArray) -> DifferentialEvaluation {
    companion object {
        fun from(epsilon: Double = 1e-8, func: (DoubleArray) -> Double): DifferentialFunction {
            return DifferentialFunction {
                val y = func.invoke(it)
                val grads = gradientsOf(
                    func,
                    it,
                    y,
                    epsilon
                )
                DifferentialEvaluation(y, grads)
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

    override fun invoke(x: DoubleArray): DifferentialEvaluation {
        return func.invoke(x)
    }

    fun negate(): DifferentialFunction {
        return DifferentialFunction { x ->
            val (y, grads) = this.invoke(x)
            DifferentialEvaluation(-y, grads.map { -it }.toDoubleArray())
        }
    }
}
