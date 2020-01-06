package thuan.handsome.gp.kernel

import koma.end
import koma.extensions.toDoubleArray
import koma.mat
import org.junit.Test
import thuan.handsome.utils.assertNearEquals

class KernelTest {
    companion object {
        val data = mat[
                0.0, 1.0, 3.0, 2.0 end
                1.0, 4.0, 4.0, 5.0 end
                5.0, 2.0, 1.0, 4.0 end
                6.0, 2.0, 1.0, 4.0 end
                5.0, 2.0, 3.0, 9.0 end
                5.0, 5.0, 1.0, 3.0
        ]
    }

    @Test
    fun testMaternTwiceDifferential() {
        val kernel = Matern(nu = MaternType.TWICE_DIFFERENTIAL)

        assertNearEquals(7.129525070484171, kernel.getCovarianceMatrix(data, theta = doubleArrayOf(0.0)).elementSum())
        assertNearEquals(
            1.6359646151176421,
            kernel.getCovarianceMatrixGradient(data, theta = doubleArrayOf(0.0)).toDoubleArray().sum()
        )

        assertNearEquals(20.603669967212486, kernel.getCovarianceMatrix(data, theta = doubleArrayOf(1.6)).elementSum())
        assertNearEquals(
            15.792818201614057,
            kernel.getCovarianceMatrixGradient(data, theta = doubleArrayOf(1.6)).toDoubleArray().sum()
        )
    }

    @Test
    fun testMaternOnceDifferential() {
        val kernel = Matern(nu = MaternType.ONCE_DIFFERENTIAL)

        assertNearEquals(7.080798824900314, kernel.getCovarianceMatrix(data, theta = doubleArrayOf(0.0)).elementSum())
        assertNearEquals(
            1.6572452375427567,
            kernel.getCovarianceMatrixGradient(data, theta = doubleArrayOf(0.0)).toDoubleArray().sum()
        )

        assertNearEquals(19.687708084799777, kernel.getCovarianceMatrix(data, theta = doubleArrayOf(1.6)).elementSum())
        assertNearEquals(
            14.432826180216908,
            kernel.getCovarianceMatrixGradient(data, theta = doubleArrayOf(1.6)).toDoubleArray().sum()
        )
    }

    @Test
    fun testMaternAbsoluteExponential() {
        val kernel = Matern(nu = MaternType.ABSOLUTE_EXPONENTIAL)

        assertNearEquals(6.971260850937075, kernel.getCovarianceMatrix(data, theta = doubleArrayOf(0.0)).elementSum())
        assertNearEquals(
            1.6667830930496677,
            kernel.getCovarianceMatrixGradient(data, theta = doubleArrayOf(0.0)).toDoubleArray().sum()
        )

        assertNearEquals(16.813394382373428, kernel.getCovarianceMatrix(data, theta = doubleArrayOf(1.6)).elementSum())
        assertNearEquals(
            10.2022297990394,
            kernel.getCovarianceMatrixGradient(data, theta = doubleArrayOf(1.6)).toDoubleArray().sum()
        )
    }
}
