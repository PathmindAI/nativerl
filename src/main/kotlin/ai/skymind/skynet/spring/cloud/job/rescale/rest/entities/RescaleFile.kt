package ai.skymind.skynet.spring.cloud.job.rescale.rest.entities

import java.time.LocalDateTime

data class RescaleFile (
        /**
         * 1 = inpute file,
         * 2 = template file,
         * 3 = parameter file,
         * 4 = script file,
         * 5 = output file,
         * 7 = design variable file,
         * 8 = case file,
         * 9 = optimizer file,
         * 10 = temporary file
         */
        val typeId: Int,
        val name: String,
        val dateUploaded: LocalDateTime,
        val relativePath: String?,
        val encodedEncryptionKey: String,
        val downloadUrl: String,
        val sharedWith: List<String>,
        val decryptedSize: Int,
        val owner: String,
        val path: String,
        val isUploaded: Boolean,
        val viewInBrowser: Boolean,
        val id: String,
        val isDeleted: Boolean,
        val md5: String
        )