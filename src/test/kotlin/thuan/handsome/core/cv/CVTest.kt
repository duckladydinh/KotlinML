package thuan.handsome.core.cv

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.jupiter.api.Test

class CVTest {
    @Test
    fun testMakeFolds() {
        for (nFolds in 2..10) {
            for (start in sequenceOf(nFolds, nFolds + 1000)) {
                for (n in start until (start + nFolds)) {
                    val folds = generateFolds(n, nFolds)
                    assertEquals(nFolds, folds.size)
                    for ((train, valid) in folds) {
                        assertTrue(train.isNotEmpty())

                        val total = train + valid
                        assertEquals(n, total.size)

                        val uniqueTotal = total.toSet()
                        assertEquals(n, uniqueTotal.size)
                        assertEquals(0, uniqueTotal.min())
                        assertEquals(n - 1, uniqueTotal.max())
                    }
                }
            }
        }
    }
}
