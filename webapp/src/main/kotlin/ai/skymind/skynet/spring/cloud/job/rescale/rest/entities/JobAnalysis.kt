package ai.skymind.skynet.spring.cloud.job.rescale.rest.entities

data class JobAnalysis (
        val command: String,
        val analysis: Analysis = Analysis("user_included", "0"),
        val hardware: Hardware = Hardware("mercury", 7),
        val useMpi: Boolean = false,
        val envVars: Map<String,String> = emptyMap(),
        val inputFiles: List<FileReference> = emptyList(),
        val useRescaleLicense: Boolean = false,
        val templateTasks: List<Any>? = emptyList(),
        val preProcessScript: String? = null,
        val preProcessScriptCommand: String = "",
        val postProcessScript: String? = null,
        val postProcessScriptCommand: String = ""
){
    data class Analysis(
            val code: String,
            val version: String? = null
    )

    data class Hardware(
            val coreType: String,
            val coresPerSlot: Int,
            val walltime: Int = 24
    )

    data class FileReference(
            val id: String,
            val decompress: Boolean = true
    )
}