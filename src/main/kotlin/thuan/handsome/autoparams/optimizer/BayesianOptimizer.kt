package thuan.handsome.autoparams.optimizer

import koma.pow
import thuan.handsome.autoparams.xspace.XSpace
import thuan.handsome.lbfgsb.LBFGSBOptimizer

class BayesianOptimizer : Optimizer {
    override fun argMaximize(
        func: (Map<String, Any>) -> Double,
        xSpace: XSpace,
        maxEvals: Int
    ): Pair<Map<String, Any>, Double> {
        TODO("not implemented") // To change body of created functions use File | Settings | File Templates.
    }

    fun bayesianMaximize(
        func: (Map<String, Any>) -> Double,
        xSpace: XSpace,
        maxEvals: Int
    ) {
    }

    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            val rosen = { params: DoubleArray ->
                val n = params.size
                var res = 0.0
                for (i in 1 until n) {
                    res += 100.0 * (params[i] - params[i - 1].pow(2)).pow(2) + (1 - params[i - 1]).pow(2)
                }

                res
            }

            val res = LBFGSBOptimizer.minimize(rosen, doubleArrayOf(1.3, 0.7, 0.8, 1.9, 1.2))
            //
            //
            println(res)
            println(rosen.invoke((0 until 10).map { it * 0.1 }.toDoubleArray()))
        }
    }
}
