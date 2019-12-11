package thuan.handsome.autoparams.env

import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.Serializable
import java.nio.file.Files

class NativeLoader(topLevelResourcesPath: String) : Serializable {
	companion object {
		val osPrefix: String
			get() {
				val osName = System.getProperty("os.name").toLowerCase()
				return if (osName.contains("linux") || osName.contains("mac") || osName.contains("darwin")) {
					""
				} else if (osName.contains("windows")) {
					"lib"
				} else {
					throw UnsatisfiedLinkError(
						String.format(
							"This component doesn't currently have native support for OS: %s",
							osName
						)
					)
				}
			}

		private fun getResourcesPath(topLevelResourcesPath: String): String {
			val sep = "/"
			val osName = System.getProperty("os.name").toLowerCase()
			val resourcePrefix = "$topLevelResourcesPath$sep%s$sep"
			return if (osName.contains("linux")) {
				String.format(resourcePrefix, "linux/x86_64")
			} else if (osName.contains("windows")) {
				String.format(resourcePrefix, "windows/x86_64")
			} else if (osName.contains("mac") || osName.contains("darwin")) {
				String.format(resourcePrefix, "osx/x86_64")
			} else {
				throw UnsatisfiedLinkError(
					String.format(
						"This component doesn't currently have native support for OS: %s",
						osName
					)
				)
			}
		}
	}

	private val tempDir = Files.createTempDirectory("tmp").toFile().apply { deleteOnExit() }
	private val resourcesPath = getResourcesPath(topLevelResourcesPath)
	private var extractionDone = false


	fun loadLibraryByName(libName: String) {
		var libraryName = libName
		try {
			System.loadLibrary(libraryName)
		} catch (e: UnsatisfiedLinkError) {
			try { // Get the OS specific library name
				libraryName = System.mapLibraryName(libraryName)
				extractNativeLibraries(libraryName)
				// Try to load library from extracted native resources
				System.load(tempDir.absolutePath + File.separator + libraryName)
			} catch (ee: Exception) {
				throw UnsatisfiedLinkError(
					String.format(
						"Could not load the native libraries because " +
								"we encountered the following problems: %s and %s",
						e.message, ee.message
					)
				)
			}
		}
	}

	private fun extractNativeLibraries(libName: String) {
		if (!extractionDone) {
			extractResourceFromPath(libName, resourcesPath)
		}
		extractionDone = true
	}

	private fun extractResourceFromPath(libName: String, prefix: String) {
		val temp = File(tempDir.path + File.separator + libName)
		temp.createNewFile()
		temp.deleteOnExit()
		if (!temp.exists()) {
			throw FileNotFoundException(
				String.format(
					"Temporary file %s could not be created. Make sure you can write to this location.",
					temp.absolutePath
				)
			)
		}
		val path = prefix + libName
		val inStream = NativeLoader::class.java.getResourceAsStream(path)
			?: throw FileNotFoundException(String.format("Could not find resource %s in jar.", path))
		val outStream = FileOutputStream(temp)
		val buffer = ByteArray(1 shl 18)
		var bytesRead: Int
		try {
			while (inStream.read(buffer).also { bytesRead = it } >= 0) {
				outStream.write(buffer, 0, bytesRead)
			}
		} finally {
			outStream.close()
			inStream.close()
		}
	}
}