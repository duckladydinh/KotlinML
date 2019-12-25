package thuan.handsome.lbfgsb

import kotlin.math.pow
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test
import thuan.handsome.utils.Bound
import thuan.handsome.utils.LOGGER

class LBFGSBTest {
    @Test
    fun testUnboundedRosen() {
        val rosen = { params: DoubleArray ->
            val n = params.size
            var res = 0.0
            for (i in 1 until n) {
                res += 100.0 * (params[i] - params[i - 1].pow(2)).pow(2) + (1 - params[i - 1]).pow(2)
            }

            res
        }
        assertEquals(76.56, rosen.invoke((0 until 10).map { it * 0.1 }.toDoubleArray()))

        val res = LBFGSBOptimizer.minimize(rosen, doubleArrayOf(1.3, 0.7, 0.8, 1.9, 1.2))
        LOGGER.info { "$res" }

        assertTrue(res.x.map { (it - 1).pow(2) }.sum() < 0.001)
        assertTrue(res.y < 0.001)
    }

    @Test
    fun testBoundedFunction() {
        val f = { params: DoubleArray ->
            val x = params[0]
            val y = params[1]
            val z = params[2]
            x * x - y * y * z
        }

        val res = LBFGSBOptimizer.minimize(
            f, doubleArrayOf(1.0, 1.0, 0.0), listOf(
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
