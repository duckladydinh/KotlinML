package thuan.handsome.core.utils

import com.google.common.flogger.FluentLogger
import com.google.common.flogger.LoggerConfig
import java.util.logging.ConsoleHandler
import java.util.logging.Level.INFO as LEVEL
import kotlin.math.roundToInt

val LOGGER = FluentLogger.forEnclosingClass()!!.apply {
    with(LoggerConfig.of(this)) {
        level = LEVEL
        addHandler(ConsoleHandler().apply {
            level = LEVEL
        })
    }
}

fun <T> T.toUncheckedInt(): Int {
    return if (this is Int) this else (this as Double).roundToInt()
}

fun <T> T.toUncheckedDouble(): Double {
    return if (this is Double) this else (this as Number).toDouble()
}
