package thuan.handsome.optimizer

import thuan.handsome.core.xspace.XSpace

interface Optimizer {
    fun argmax(func: (Map<String, Any>) -> Double, xSpace: XSpace, maxiter: Int): Pair<Map<String, Any>, Double>
}
