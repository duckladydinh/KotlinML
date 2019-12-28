package thuan.handsome.core.utils

import kotlin.math.roundToInt
import mu.KotlinLogging

val LOGGER = KotlinLogging.logger {}

fun <T> T.toUncheckedInt(): Int {
    return if (this is Int) this else (this as Double).roundToInt()
}

fun <T> T.toUncheckedDouble(): Double {
    return if (this is Double) this else (this as Number).toDouble()
}
