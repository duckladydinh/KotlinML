package thuan.handsome.core.utils

import com.google.common.flogger.FluentLogger
import com.google.common.flogger.LoggerConfig
import java.util.logging.ConsoleHandler
import java.util.logging.Level.INFO as LEVEL

class Logger {
    companion object {
        private val FLOGGER = FluentLogger.forEnclosingClass()!!

        init {
            System.setProperty("java.util.logging.SimpleFormatter.format", "%4\$s: %5\$s%n")

            with(LoggerConfig.of(FLOGGER)) {
                level = LEVEL
                addHandler(ConsoleHandler().apply {
                    level = LEVEL
                })
            }
        }

        fun warning(): FluentLogger.Api {
            return FLOGGER.atWarning()
        }

        fun fine(): FluentLogger.Api {
            return FLOGGER.atFine()
        }
    }
}
