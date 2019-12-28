package thuan.handsome.gp.kernel

enum class MaternType(val nu: Double) {
    ABSOLUTE_EXPONENTIAL(0.5),
    ONCE_DIFFERENTIAL(1.5),
    TWICE_DIFFERENTIAL(2.5)
}
