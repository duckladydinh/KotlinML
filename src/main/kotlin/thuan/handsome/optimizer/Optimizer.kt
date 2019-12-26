package thuan.handsome.optimizer

import thuan.handsome.core.xspace.XSpace

interface Optimizer {
    fun argMaximize(
        func: (Map<String, Any>) -> Double,
        xSpace: XSpace,
        maxEvals: Int
    ): Pair<Map<String, Any>, Double>
}
