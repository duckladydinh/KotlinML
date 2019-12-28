package thuan.handsome.lightgbm

import thuan.handsome.core.env.NativeLoader

abstract class CObject protected constructor() : AutoCloseable {
    val handle: PP_VOID = C_API.new_voidpp()

    companion object {
        init {
            val osPrefix = NativeLoader.osPrefix
            NativeLoader("/com/microsoft/ml/lightgbm").loadLibraryByName(osPrefix + "_lightgbm")
            NativeLoader("/com/microsoft/ml/lightgbm").loadLibraryByName(osPrefix + "_lightgbm_swig")
        }
    }
}
