package thuan.handsome.lbfgsb

enum class StopType {
	/** algorithm ended by reaching the maximum iterations limit  */
	MAX_ITERATIONS,
	/** algorithm ended by satisfying one of the other stop conditions */
	CONVERGENT,
	/** algorithm ended but wasn't able to satisfy the stop conditions */
	ABNORMAL,
	/** algorithm stopped by the user  */
	USER
}
