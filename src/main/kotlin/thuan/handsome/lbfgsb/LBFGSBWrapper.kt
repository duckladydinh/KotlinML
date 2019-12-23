package thuan.handsome.lbfgsb

import thuan.handsome.lbfgsb.jni.*
import thuan.handsome.utils.NativeLoader

internal class LBFGSBWrapper(private val dimensions: Int, numCorrections: Int) {
    private val data: lbfgsb = lbfgsb_wrapper.lbfgsb_create(dimensions, numCorrections)

    companion object {
        init {
            NativeLoader("/thuan/handsome/lbfgsb").loadLibraryByName("lbfgsb_wrapper")
        }

        private fun getBoundCode(bound: Bound): Int {
            if (!bound.isLowerBoundDefined() && !bound.isUpperBoundDefined()) return 0
            if (bound.isLowerBoundDefined() && !bound.isUpperBoundDefined()) return 1
            if (bound.isLowerBoundDefined() && bound.isUpperBoundDefined()) return 2
            return 3
        }

        private fun javaArrayToNative(javaArray: DoubleArray, nativeArray: SWIGTYPE_p_double) {
            for (i in javaArray.indices) {
                lbfgsb_wrapper.doubleArray_setitem(nativeArray, i, javaArray[i])
            }
        }

        private fun nativeArrayToJava(nativeArray: SWIGTYPE_p_double, length: Int): DoubleArray {
            val javaArray = DoubleArray(length)
            for (i in javaArray.indices) {
                javaArray[i] = lbfgsb_wrapper.doubleArray_getitem(nativeArray, i)
            }
            return javaArray
        }
    }

    fun setBounds(bounds: List<Bound>) {
        assert(dimensions == bounds.size)
        val nbd = data.nbd
        val l = data.l
        val u = data.u

        for ((index, bound) in bounds.withIndex()) {
            lbfgsb_wrapper.intArray_setitem(nbd, index, getBoundCode(bound))
            lbfgsb_wrapper.doubleArray_setitem(l, index, bound.lower ?: Double.NEGATIVE_INFINITY)
            lbfgsb_wrapper.doubleArray_setitem(u, index, bound.upper ?: Double.POSITIVE_INFINITY)
        }
    }

    fun setFunctionFactor(value: Double) {
        data.factr = value
    }

    fun setMaxGradientNorm(value: Double) {
        data.pgtol = value
    }

    fun getX(): DoubleArray {
        return nativeArrayToJava(data.x, dimensions)
    }

    fun setX(x: DoubleArray) {
        javaArrayToNative(x, data.x)
    }

    private fun getTask(): lbfgsb_task_type {
        return lbfgsb_wrapper.lbfgsb_get_task(data)
    }

    private fun setTask(type: lbfgsb_task_type) {
        lbfgsb_wrapper.lbfgsb_set_task(data, type)
    }

    fun getY(): Double {
        return data.f
    }

    private fun setY(y: Double) {
        data.f = y
    }

    fun getGrads(): DoubleArray {
        return nativeArrayToJava(data.g, dimensions)
    }

    private fun setGrads(grads: DoubleArray) {
        javaArrayToNative(grads, data.g)
    }

    private fun getStateDescription(): String {
        return data.task.trim { it <= ' ' }
    }

    fun setDebugLevel(debugLevel: Int) {
        data.iprint = debugLevel
    }

    private fun step() {
        lbfgsb_wrapper.lbfgsb_step(data)
    }

    fun minimize(
        func: (DoubleArray) -> Pair<Double, DoubleArray>,
        maxIterations: Int,
        onIterationFinished: ((DoubleArray, Double, DoubleArray) -> Boolean)?
    ): StopInfo {
        setTask(lbfgsb_task_type.LBFGSB_START)
        var numEvals = 0
        for (iteration in 1..maxIterations) {
            step()

            val task = getTask()
            assert(task != lbfgsb_task_type.LBFGSB_ERROR) { "L-BFGS-B Error: ${getStateDescription()}" }

            when (task) {
                lbfgsb_task_type.LBFGSB_FG -> {
                    numEvals += 1
                    val x = getX()
                    val evaluations = func.invoke(x)
                    setY(evaluations.first)
                    setGrads(evaluations.second)
                }

                lbfgsb_task_type.LBFGSB_NEW_X -> {
                    if (onIterationFinished != null) {
                        val isContinue = onIterationFinished(getX(), getY(), getGrads())
                        if (!isContinue) {
                            stopAlgorithm()
                            return StopInfo(iteration, numEvals, StopType.USER, getStateDescription())
                        }
                    }
                }

                lbfgsb_task_type.LBFGSB_CONV -> return StopInfo(
                    iteration, numEvals, StopType.CONVERGENT, getStateDescription()
                )

                lbfgsb_task_type.LBFGSB_ABNO -> return StopInfo(
                    iteration, numEvals, StopType.ABNORMAL, getStateDescription()
                )

                else -> throw RuntimeException("L-BFGS-B: Unknown task")
            }
        }

        stopAlgorithm()
        return StopInfo(maxIterations, numEvals, StopType.MAX_ITERATIONS, getStateDescription())
    }

    fun close() {
        lbfgsb_wrapper.lbfgsb_delete(data)
    }

    private fun stopAlgorithm() {
        setTask(lbfgsb_task_type.LBFGSB_STOP)
        step()
    }
}
