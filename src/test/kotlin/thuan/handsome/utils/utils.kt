package thuan.handsome.utils

import kotlin.math.abs
import kotlin.test.assertTrue

const val EPS = 1e-9

fun <A : Number, B : Number> assertNearEquals(a: A, b: B) {
    assertTrue(abs(a.toDouble() - b.toDouble()) < EPS, "$a and $b are not nearly equal!")
}
