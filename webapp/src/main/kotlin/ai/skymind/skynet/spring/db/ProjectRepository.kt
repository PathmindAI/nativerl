package ai.skymind.skynet.spring.db

import ai.skymind.skynet.data.db.jooq.Tables
import ai.skymind.skynet.data.db.jooq.tables.records.ProjectRecord
import org.jooq.DSLContext
import org.springframework.stereotype.Service

@Service
class ProjectRepository(
        val ctx: DSLContext
){
    private fun all(ownerId: Int) = ctx.selectFrom(Tables.PROJECT).where(Tables.PROJECT.USER_ID.eq(ownerId))
    fun findById(ownerId: Int, projectId: Int?): ProjectRecord? = all(ownerId).and(Tables.PROJECT.ID.eq(projectId)).fetchOne()
    fun find(ownerId: Int, query: String?) = if(query.isNullOrBlank()) findAll(ownerId) else all(ownerId).and(Tables.PROJECT.NAME.containsIgnoreCase(query)).fetch().toList()

    fun findAll(ownerId: Int) = all(ownerId).fetch().toList()

    fun add(project: ProjectRecord): ProjectRecord {
        project.attach(ctx.configuration())
        project.store()
        return project
    }

}