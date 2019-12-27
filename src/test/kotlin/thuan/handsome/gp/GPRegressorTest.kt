package thuan.handsome.gp

import koma.*
import koma.extensions.reshape
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test
import thuan.handsome.core.utils.LOGGER

class GPRegressorTest {
    companion object {
        const val EPS = 1e-9
    }

    @Test
    fun logMarginalLikelihoodTest() {
        val data = create(doubleArrayOf(1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)).reshape(6, 2)
        val y = create(doubleArrayOf(0.84147098, -4.79462137, 4.59890619, -1.67649299, -3.02720998, 1.81859485)).T
        val gp = GPRegressor(data, y)
        val (likelihood, grads) = gp.evaluate(doubleArrayOf(0.001), true)
        assertEquals(-34.43188191091343, likelihood)
        assertEquals(-2.2729470072358002, grads[0])
    }

    @Test
    fun fitAndPredictionTest() {
        val data = create(doubleArrayOf(1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)).reshape(6, 2)
        val y = create(doubleArrayOf(0.84147098, -4.79462137, 4.59890619, -1.67649299, -3.02720998, 1.81859485)).T
        val gp = GPRegressor.fit(data, y, numOptimizerRestarts = 2)

        val res = gp.evaluate(gp.bestTheta, computeGradient = true)
        LOGGER.info { res }

        val (likelihood, _) = res
        assertTrue(likelihood >= -35)

        val (mean, variance) = gp.predict(doubleArrayOf(2.5, 3.5))
        assertTrue(mean >= -0.5)
    }

    // mean to be disabled
    @Test
    fun weirdTest() {
        val data = mat[ 0.2257546588449, 0.98125148573425, 28.88995549454187, 488.34183075465603, 0.65236391569299, 5.86857442322238 end
                0.86553265379434, 0.18613563397492, 44.32598823292256, 211.36703053295125, 0.86686782251728, 6.79226526666322 end
                0.73719011577892, 0.75535955654509, 24.91990726818348, 304.6294419271865, 0.65089476774898, 2.03685567596685 end
                0.74035082273829, 0.06462440982132, 27.37217592993344, 348.38408185707345, 0.70527894053902, 7.83801954878825 end
                0.01108910280217, 0.39852920004714, 14.55355618319405, 392.4984859545136, 0.31098375779818, 6.95207456174469 ]
        val y = mat[ 0.65188470066519, 0.94640122511485, 0.9578313253012, 0.94060995184591, 0.61241970021413 ].T
        val gp = GPRegressor.fit(data, y, 50)
        val res = gp.evaluate(gp.bestTheta, computeGradient = true)
        LOGGER.info { res }
    }
}
