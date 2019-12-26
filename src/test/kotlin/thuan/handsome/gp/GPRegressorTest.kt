package thuan.handsome.gp

import koma.create
import koma.extensions.reshape
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class GPRegressorTest {
    companion object {
        const val EPS = 1e-9
    }

    @Test
    fun logMarginalLikelihoodTest() {
        val data = create(doubleArrayOf(1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)).reshape(6, 2)
        val y = create(doubleArrayOf(0.84147098, -4.79462137, 4.59890619, -1.67649299, -3.02720998, 1.81859485)).T
        val gp = GPRegressor(data, y)
        val (likelihood, grads) = gp.evaluate(doubleArrayOf(0.001, 0.001), true)
        assertEquals(-34.43188191091343, likelihood)
        assertEquals(listOf(-2.2729470072358002, -0.8526608012597641), grads.toList())
    }

    @Test
    fun fitAndPredictionTest() {
        val data = create(doubleArrayOf(1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)).reshape(6, 2)
        val y = create(doubleArrayOf(0.84147098, -4.79462137, 4.59890619, -1.67649299, -3.02720998, 1.81859485)).T
        val gp = GPRegressor.fit(data, y, numOptimizerRestarts = 2)

        val (likelihood, grads) = gp.evaluate(gp.theta, computeGradient = true)
        assertTrue(abs(listOf(3.275441966932302E-6, 6.448424168213618E-7).sum() - grads.sum()) <= EPS)
        assertTrue(abs(-32.04890772304928 - likelihood) < EPS)

        gp.evaluate(doubleArrayOf(-0.10697528, -0.10697528), computeGradient = true)
        val (mean, variance) = gp.predict(doubleArrayOf(2.5, 3.5))
        assertTrue(abs(0.909870822511425 - variance) < EPS)
        assertTrue(abs(-0.4503533628761439 - mean) < EPS)
    }
}
