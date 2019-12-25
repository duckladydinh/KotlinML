package thuan.handsome.gp

import koma.create
import koma.extensions.reshape
import kotlin.test.assertEquals
import org.junit.Test

class GPRegressorTest {
    @Test
    fun logMarginalLikelihoodTest() {
        val data = create(doubleArrayOf(1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)).reshape(6, 2)
        val y = create(doubleArrayOf(0.84147098, -4.79462137, 4.59890619, -1.67649299, -3.02720998, 1.81859485)).T
        val gp = GPRegressor(data, y)
        gp.updateTheta(doubleArrayOf(0.001, 0.001), true)
        assertEquals(-34.43188191091343, gp.logMarginalLikelihood)
        assertEquals(listOf(-2.2729470072358002, -0.8526608012597641), gp.logMarginalLikelihoodGrads.toList())
    }
}
