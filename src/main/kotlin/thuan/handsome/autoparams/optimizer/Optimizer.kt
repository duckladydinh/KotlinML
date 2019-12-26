package thuan.handsome.autoparams.optimizer

import thuan.handsome.ml.xspace.XSpace

interface Optimizer {
    fun argMaximize(
        func: (Map<String, Any>) -> Double,
        xSpace: XSpace,
        maxEvals: Int
    ): Pair<Map<String, Any>, Double>
}
