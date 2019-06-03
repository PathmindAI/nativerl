package ai.skymind.skynet.spring.cloud.job.rescale.rest.entities

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import java.time.LocalDateTime

@JsonIgnoreProperties(ignoreUnknown = true)
data class RescaleFile (
        /**
         * 1 = inpute file,
         * 2 = template file,
         * 3 = parameter file,
         * 4 = script file,
         * 5 = output file,
         * 7 = design variable file,
         * 8 = case fvile,
         * 9 = optimizer file,
         * 10 = temporary file
         */
        val typeId: Int,
        val id: String,
        val name: String,
        val isUploaded: Boolean,
        val isDeleted: Boolean,
        val viewInBrowser: Boolean,
        val dateUploaded: LocalDateTime,
        val relativePath: String?,
        val downloadUrl: String,
        val path: String,
        val sharedWith: List<String>,
        val owner: String,
        val encodedEncryptionKey: String,
        val decryptedSize: Int,
        val md5: String
        )