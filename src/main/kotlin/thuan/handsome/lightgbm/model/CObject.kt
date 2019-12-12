package thuan.handsome.lightgbm.model

import java.io.Closeable

abstract class CObject : Closeable {
	val handle: VoidDoublePointer = API.new_voidpp()

	companion object {
		init {
			initNative()
		}
	}
}