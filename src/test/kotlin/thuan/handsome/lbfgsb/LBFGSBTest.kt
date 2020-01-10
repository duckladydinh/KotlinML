package thuan.handsome.lbfgsb

import kotlin.math.pow
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.jupiter.api.Test
import thuan.handsome.core.function.DifferentialFunction
import thuan.handsome.core.utils.LOGGER
import thuan.handsome.core.xspace.Bound

class LBFGSBTest {
    @Test
    fun testUnboundedRosen() {
        val rosen = DifferentialFunction.from {
            val n = it.size
            var res = 0.0
            for (i in 1 until n) {
                res += 100.0 * (it[i] - it[i - 1].pow(2)).pow(2) + (1 - it[i - 1]).pow(2)
            }
            res
        }
        val (y, _) = rosen.invoke((0 until 10).map { it * 0.1 }.toDoubleArray())
        assertEquals(76.56, y)

        val res = LBFGSBWrapper.minimize(rosen, doubleArrayOf(1.3, 0.7, 0.8, 1.9, 1.2))
        LOGGER.info { "$res" }

        assertTrue(res.x.map { (it - 1).pow(2) }.sum() < 0.001)
        assertTrue(res.y < 0.001)
    }

    @Test
    fun testBoundedFunction() {
        val res = LBFGSBWrapper.minimize(
            DifferentialFunction.from {
                it[0].pow(2) - it[1].pow(2) * it[2]
            },
            doubleArrayOf(1.0, 1.0, 0.0), arrayOf(
                Bound(1.0, 5.0),
                Bound(-2.0, 3.0),
                Bound(-5.0, 1.0)
            )
        )
        LOGGER.info { "$res" }

        assertEquals(doubleArrayOf(1.0, 3.0, 1.0).toList(), res.x.toList())
        assertEquals(-8.0, res.y)
    }
}
