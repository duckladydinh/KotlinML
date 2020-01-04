package thuan.handsome.optimizer

import kotlin.math.pow
import org.junit.Test
import thuan.handsome.core.xspace.UniformXSpace
import thuan.handsome.gp.kernel.*

class NaiveOptimizerTest {
    companion object {
        fun testNaiveOptimizer(optimizer: Optimizer) {
            val xSpace = UniformXSpace()
            xSpace.addParam("x", -100.0, 100.0)
            xSpace.addParam("y", -100.0, 100.0)
            xSpace.addParam("z", -100.0, 100.0)
            xSpace.addParam("w", -100.0, 100.0)
            val func = fun(params: Map<String, Any>): Double {
                val x = params["x"] as Double
                val y = params["y"] as Double
                val z = params["z"] as Double
                val w = params["w"] as Double
                return -((x - 1).pow(2) + (y - 2).pow(2) + (z - 3).pow(2) + (w - 4).pow(2))
            }
            val (x, y) = optimizer.argmax(func, xSpace = xSpace, maxiter = 30)
            println(x.map { "${it.key} : ${it.value}" }.joinToString(" "))
            println(y)
        }
    }
    @Test
    fun naiveFunctionTestWithRBF() {
        testNaiveOptimizer(BayesianOptimizer(kernel = RBF()))
    }

    @Test
    fun naiveFunctionTestWithMatern25() {
        testNaiveOptimizer(BayesianOptimizer(kernel = Matern(nu = MaternType.TWICE_DIFFERENTIAL)))
    }

    @Test
    fun naiveFunctionTestWithMatern15() {
        testNaiveOptimizer(BayesianOptimizer(kernel = Matern(nu = MaternType.ONCE_DIFFERENTIAL)))
    }

    @Test
    fun naiveFunctionTestWithMatern05() {
        testNaiveOptimizer(BayesianOptimizer(kernel = Matern(nu = MaternType.ABSOLUTE_EXPONENTIAL)))
    }

    @Test
    fun naiveFunctionTestWithRandom() {
        testNaiveOptimizer(UniformOptimizer())
    }
}
