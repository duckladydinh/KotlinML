package thuan.handsome.utils

import thuan.handsome.core.xspace.*

private val xSpace = UniformXSpace().apply {
    addConstantParams(
        mapOf(
            "objective" to "binary",
            "is_unbalance" to false,
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
