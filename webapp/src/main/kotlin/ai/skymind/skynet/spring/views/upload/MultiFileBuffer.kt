package ai.skymind.skynet.spring.views.upload

import com.vaadin.external.org.slf4j.LoggerFactory
import com.vaadin.flow.component.upload.MultiFileReceiver
import com.vaadin.flow.component.upload.receivers.FileData
import java.io.*
import java.util.*

class MultiFileBuffer : MultiFileReceiver {
    private val files = HashMap<String, FileData>()
    private val tempFileNames = HashMap<String, String>()

    private fun createFileOutputStream(fileName: String): FileOutputStream? {
        try {
            return FileOutputStream(createFile(fileName))
        } catch (e: IOException) {
            LOGGER.warn("Failed to create file output stream for: '$fileName'", e)
        }

        return null
    }

    private fun createFile(fileName: String): File {
        val tempFileName = ("upload_tmpfile_" + fileName + "_"
                + System.currentTimeMillis())

        val tempFile = File.createTempFile(tempFileName, null)
        tempFileNames[fileName] = tempFile.path

        return tempFile
    }

    override fun receiveUpload(fileName: String, MIMEType: String): OutputStream? {
        val outputBuffer = createFileOutputStream(fileName)
        files[fileName] = FileData(fileName, MIMEType, outputBuffer)

        return outputBuffer
    }

    fun getFiles(): Set<String> {
        return files.keys
    }

    fun getFileData(fileName: String): FileData? {
        return files[fileName]
    }


    fun getInputStream(fileName: String): InputStream {
        if (tempFileNames.containsKey(fileName)) {
            try {
                return FileInputStream(tempFileNames[fileName])
            } catch (e: IOException) {
                LOGGER.warn("Failed to create InputStream for: '$fileName'", e)
            }

        }
        return ByteArrayInputStream(ByteArray(0))
    }

    companion object {
        private val LOGGER = LoggerFactory.getLogger(MultiFileBuffer::class.java)
    }

}