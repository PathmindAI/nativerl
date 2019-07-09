package ai.skymind.skynet.spring.services

import ai.skymind.skynet.data.db.jooq.Tables
import ai.skymind.skynet.data.db.jooq.tables.records.ProjectRecord
import ai.skymind.skynet.spring.cloud.job.api.JobExecutor
import ai.skymind.skynet.spring.db.ModelRepository
import ai.skymind.skynet.spring.db.ProjectRepository
import org.springframework.stereotype.Service
import java.io.File

@Service
class ProjectService(
        val projectRepository: ProjectRepository,
        val modelRepository: ModelRepository,
        val cloudJobExecutor: JobExecutor
) {
    fun find(ownerId: Int, query: String?) = projectRepository.find(ownerId, query)
    fun findAll(ownerId: Int) = projectRepository.findAll(ownerId)

    fun addProject(ownerId: Int, projectName: String, model: File, modelTimeUnit: String, modelStepSize: Int): ProjectRecord {
        val project = Tables.PROJECT.newRecord().apply {
            name = projectName
            userId = ownerId
        }.let {
            projectRepository.add(it)
        }

        val modelFileId = cloudJobExecutor.upload(model)
        Tables.MODEL.newRecord().apply {
            name = "Initial Model for Project '$projectName'"
            userId = ownerId
            projectId = project.id
            fileId = modelFileId
            timeUnit = modelTimeUnit
            stepSize = modelStepSize
        }.let {
            modelRepository.add(it)
        }
        return project
    }

    fun models(projectId: Int) = modelRepository.findAllByProjectId(projectId)
    fun findById(ownerId: Int, projectId: Int?) = projectRepository.findById(ownerId, projectId)
}