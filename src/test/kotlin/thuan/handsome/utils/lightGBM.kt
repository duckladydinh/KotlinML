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
    addParam("feature_fraction", 1e-9, 1.0 - 1e-9)
    addParam("bagging_fraction", 1e-9, 1.0 - 1e-9)
    addParam("num_leaves", 15.0, 127.0, XType.INT)
    addParam("min_split_gain", 1e-9, 1.0)
    addParam("min_child_weight", 1e-9, 1.0)
}

fun getTestXSpace(): XSpace {
    return xSpace
}
