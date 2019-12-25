package thuan.handsome.autoparams.optimizer

import thuan.handsome.autoparams.xspace.XSpace

class BayesianOptimizer : Optimizer {
    override fun argMaximize(
        func: (Map<String, Any>) -> Double,
        xSpace: XSpace,
        maxEvals: Int
    ): Pair<Map<String, Any>, Double> {
        TODO("not implemented") // To change body of created functions use File | Settings | File Templates.
    }
}
