package ai.skymind.skynet.spring.services

import ai.skymind.skynet.data.db.jooq.Tables
import ai.skymind.skynet.data.db.jooq.tables.records.ProjectRecord
import ai.skymind.skynet.spring.cloud.job.api.CloudJobExecutor
import ai.skymind.skynet.spring.db.ModelRepository
import ai.skymind.skynet.spring.db.ProjectRepository
import org.springframework.stereotype.Service
import java.io.File

@Service
class ProjectService(
        val projectRepository: ProjectRepository,
        val modelRepository: ModelRepository,
        val cloudJobExecutor: CloudJobExecutor
) {
    fun find(ownerId: Int, query: String?) = projectRepository.find(ownerId, query)
    fun findAll(ownerId: Int) = projectRepository.findAll(ownerId)

    fun addProject(ownerId: Int, projectName: String, model: File): ProjectRecord {
        val project = Tables.PROJECT.newRecord().apply {
            name = projectName
            userId = ownerId
        }.let {
            projectRepository.add(it)
        }

        val modelFileId = cloudJobExecutor.upload(model)
        val model = Tables.MODEL.newRecord().apply {
            name = "Initial Model for Project '$projectName'"
            userId = ownerId
            projectId = project.id
            fileId = modelFileId
        }.let {
            modelRepository.add(it)
        }
        return project
    }

    fun models(projectId: Int) = modelRepository.findAllByProjectId(projectId)
}