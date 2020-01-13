package thuan.handsome.optimizer

import kotlin.math.pow
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource
import thuan.handsome.core.xspace.UniformXSpace
import thuan.handsome.gp.kernel.*

class MathFuncTest {
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
            println("At { ${(x.map { "${it.key} : ${it.value}" }.joinToString(" | "))} }")
            println("y = $y")
        }
    }

    @Test
    fun naiveFunctionTestWithRBF() {
        testNaiveOptimizer(BayesianOptimizer(kernel = RBF()))
    }

    @ParameterizedTest
    @EnumSource(MaternType::class)
    fun naiveFunctionTestWithMatern(maternType: MaternType) {
        testNaiveOptimizer(BayesianOptimizer(kernel = Matern(nu = maternType)))
    }

    @Test
    fun naiveFunctionTestWithRandom() {
        testNaiveOptimizer(UniformOptimizer())
    }
}
