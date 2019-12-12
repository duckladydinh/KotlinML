package thuan.handsome.lightgbm.metric

import kotlin.test.assertEquals

fun f1score(predicted: IntArray, target: IntArray): Double {
	assertEquals(predicted.size, target.size)
	// TN, TP, FN, FP
	val stats = intArrayOf(0, 0, 0, 0)
	val n = predicted.size

	repeat(n) {
		if (target[it] == predicted[it]) {
			stats[predicted[it]] += 1
		} else {
			stats[predicted[it] + 2] += 1
		}
	}

	if (stats[0] == n) {
		return 1.0
	}
	if (stats[1] == 0) {
		return 0.0
	}

	val precision = stats[1].toDouble() / (stats[1] + stats[3])
	val recall = stats[1].toDouble() / (stats[1] + stats[2])
	return 2 * (precision * recall) / (precision + recall)
}

