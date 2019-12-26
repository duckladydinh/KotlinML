package thuan.handsome.lbfgsb

import thuan.handsome.core.function.DifferentialFunction
import thuan.handsome.core.utils.NativeLoader
import thuan.handsome.core.xspace.Bound
import thuan.handsome.lbfgsb.jni.*

class CWrapper private constructor(private val dimensions: Int, numCorrections: Int) {
    private val data: lbfgsb = lbfgsb_wrapper.lbfgsb_create(dimensions, numCorrections)

    companion object {
        init {
            NativeLoader("/thuan/handsome/lbfgsb").loadLibraryByName("lbfgsb_wrapper")
        }

        /**
		 * @param func - a differential function
		 *
		 * @param xZero - an initial guess
		 *
		 * @param bounds - an array of Bounds which defines the limits of each variable.
		 * Null value is used for no limit such as Bound(1, null) means no upper limit.
		 *
		 * @param maxGradientNorm - (pgtol) maximal acceptable gradient value
		 * The iteration will stop when ``max{|proj g_i | i = 1, ..., n} <= pgtol``
		 * where $proj g_i$ is i-th component of the the projected gradient.
		 *
		 * @param maxCorrections - The maximum number of variable metric corrections used
		 * to define the limited memory matrix. (The limited memory BFGS method does
		 * not store the full hessian but uses this many terms in an approximation to it).
		 *
		 * According to the original fortran documentation, range [3, 20] is recommended.
		 *
		 * @param funcReductionFactor - (factr, ftol/eps) relative function reduction
		 * factor. The iteration will stop when
		 *              `(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``,
		 * where ``eps`` is the machine precision, which is automatically
		 * generate`(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``,
		 * where ``eps`` is the machine precision, which is automatically
		 * generated by the coded by the code.
		 *
		 * Example values for 15 digits accuracy:
		 *  * 1e+12 for low accuracy,
		 *  * 1e+7  for moderate accuracy,
		 *  * 1e+1  for extremely high accuracy
		 *
		 *  @param verbose -1 means no logging, increasing will increase logging
		 *
		 *  @param callback (x, y, grads) - a method to call after each iteration
		 */
        fun minimize(
            // most important inputs
            func: DifferentialFunction,
            xZero: DoubleArray,
            bounds: Array<Bound> = Bound().times(xZero.size),
            // algorithm's parameters
            maxIterations: Int = 15000,
            maxCorrections: Int = 10,
            maxGradientNorm: Double = 1e-5,
            funcReductionFactor: Double = 1e7,
            // external parameters
            verbose: Int = -1,
            callback: ((DoubleArray, Double, DoubleArray) -> Boolean)? = null
        ): Summary {
            assert(bounds.size == xZero.size) {
                "Bounds number (${bounds.size}) doesn't match starting point size (${xZero.size})"
            }

            val optimizer = CWrapper(xZero.size, maxCorrections).apply {
                setX(xZero)
                setBounds(bounds)
                setDebugLevel(verbose)
                setMaxGradientNorm(maxGradientNorm)
                setFunctionFactor(funcReductionFactor)
            }

            val summary = optimizer.minimize(func, maxIterations, callback)

            optimizer.close()
            return summary
        }

        private fun getBoundCode(bound: Bound): Int {
            if (!bound.isLowerBounded() && !bound.isUpperBounded()) return 0
            if (bound.isLowerBounded() && !bound.isUpperBounded()) return 1
            if (bound.isLowerBounded() && bound.isUpperBounded()) return 2
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

    private fun minimize(
        func: DifferentialFunction,
        maxIterations: Int,
        onIterationFinished: ((DoubleArray, Double, DoubleArray) -> Boolean)?
    ): Summary {
        setTask(lbfgsb_task_type.LBFGSB_START)
        var stopType = StopType.MAX_ITERATIONS
        var numEvals = 0

        LOOP@ for (iteration in 1..maxIterations) {
            step()

            val task = getTask()
            assert(task != lbfgsb_task_type.LBFGSB_ERROR) {
                "L-BFGS-B Error: ${getState()}"
            }

            when (task) {
                lbfgsb_task_type.LBFGSB_FG -> {
                    numEvals += 1
                    val x = getX()
                    val (y, grads) = func.invoke(x)
                    setY(y)
                    setGrads(grads)
                }

                lbfgsb_task_type.LBFGSB_NEW_X -> {
                    if (onIterationFinished != null) {
                        val isContinue = onIterationFinished(getX(), getY(), getGrads())
                        if (!isContinue) {
                            stop()
                            stopType = StopType.USER
                        }
                    }
                }

                lbfgsb_task_type.LBFGSB_CONV -> {
                    stopType = StopType.CONVERGENT
                }

                lbfgsb_task_type.LBFGSB_ABNO -> {
                    stopType = StopType.ABNORMAL
                }

                else -> throw RuntimeException("L-BFGS-B: Unknown task")
            }
            if (stopType != StopType.MAX_ITERATIONS) {
                break
            }
        }
        stop()

        return Summary(this.getX(), this.getY(), this.getGrads(), maxIterations, numEvals, stopType, getState())
    }

    private fun step() {
        lbfgsb_wrapper.lbfgsb_step(data)
    }

    private fun getY(): Double {
        return data.f
    }

    private fun setY(y: Double) {
        data.f = y
    }

    private fun getX(): DoubleArray {
        return nativeArrayToJava(data.x, dimensions)
    }

    private fun setX(x: DoubleArray) {
        javaArrayToNative(x, data.x)
    }

    private fun getGrads(): DoubleArray {
        return nativeArrayToJava(data.g, dimensions)
    }

    private fun setGrads(grads: DoubleArray) {
        javaArrayToNative(grads, data.g)
    }

    private fun setBounds(bounds: Array<Bound>) {
        assert(dimensions == bounds.size)
        val nbd = data.nbd
        val l = data.l
        val u = data.u

        for ((index, bound) in bounds.withIndex()) {
            lbfgsb_wrapper.intArray_setitem(nbd, index, getBoundCode(bound))
            lbfgsb_wrapper.doubleArray_setitem(l, index, bound.lower)
            lbfgsb_wrapper.doubleArray_setitem(u, index, bound.upper)
        }
    }

    private fun setDebugLevel(debugLevel: Int) {
        data.iprint = debugLevel
    }

    private fun setFunctionFactor(value: Double) {
        data.factr = value
    }

    private fun setMaxGradientNorm(value: Double) {
        data.pgtol = value
    }

    private fun getTask(): lbfgsb_task_type {
        return lbfgsb_wrapper.lbfgsb_get_task(data)
    }

    private fun setTask(type: lbfgsb_task_type) {
        lbfgsb_wrapper.lbfgsb_set_task(data, type)
    }

    private fun getState(): String {
        return data.task.trim { it <= ' ' }
    }

    private fun stop() {
        setTask(lbfgsb_task_type.LBFGSB_STOP)
        step()
    }

    private fun close() {
        lbfgsb_wrapper.lbfgsb_delete(data)
    }
}
