package ai.skymind.skynet.spring.cloud.job.rescale.util

import java.io.File
import java.io.InputStream
import java.io.StringWriter
import java.util.concurrent.TimeUnit

data class ProcessResult(val exitValue: Int, val stdOut: String, val stdErr: String)

fun List<String>.runCommand(workingDir: File): ProcessResult {
    val proc = ProcessBuilder(*this.toTypedArray())
            .directory(workingDir)
            .redirectOutput(ProcessBuilder.Redirect.PIPE)
            .redirectError(ProcessBuilder.Redirect.PIPE)
            .start()

    val stdOut = StringWriter()
    val stdErr = StringWriter()

    do {
        proc.waitFor(1, TimeUnit.SECONDS)
        stdOut.writeAvailable(proc.inputStream)
        stdErr.writeAvailable(proc.errorStream)
    }while(proc.isAlive)

    val exitValue = proc.exitValue()

    stdOut.writeAvailable(proc.inputStream)
    stdErr.writeAvailable(proc.errorStream)

    return ProcessResult(exitValue, stdOut.toString(), stdErr.toString())
}

fun String.runCommand(workingDir: File): ProcessResult {
    val parts = this.split("\\s".toRegex())
    return parts.runCommand(workingDir)
}

fun StringWriter.writeAvailable(inputStream: InputStream) {
    while(inputStream.available() > 0){
        this.write(inputStream.read())
    }
}