package thuan.handsome.optimizer.numeric

import thuan.handsome.core.function.DifferentialFunction
import thuan.handsome.core.xspace.Bound
import thuan.handsome.lbfgsb.LBFGSBWrapper

interface NumericOptimizer {
    companion object {
        @JvmStatic
        fun maximize(
            func: DifferentialFunction,
            xZero: DoubleArray,
            bounds: Array<Bound>,
            maxiter: Int = 15000,
            type: OptimizerType = OptimizerType.L_BFGS_B
        ): XYPoint {
            when (type) {
                OptimizerType.L_BFGS_B -> {
                    val negatedFunc = func.negate()
                    val summary = LBFGSBWrapper.minimize(negatedFunc, xZero, bounds, maxiter)
                    return XYPoint(summary.x, -summary.y)
                }
                else -> throw UnsupportedOperationException("Optimizer type '$type' is unsupported!")
            }
        }
    }
}
