package ai.skymind.skynet.spring.db

import ai.skymind.skynet.data.db.jooq.Tables
import ai.skymind.skynet.data.db.jooq.tables.records.ModelRecord
import org.jooq.DSLContext
import org.springframework.stereotype.Service

@Service
class ModelRepository(
        val ctx: DSLContext
) {
    fun findAllByProjectId(projectId: Int) = ctx.selectFrom(Tables.MODEL).where(Tables.MODEL.PROJECT_ID.eq(projectId)).fetch().toList()
    fun add(it: ModelRecord): ModelRecord {
        it.attach(ctx.configuration())
        it.store()
        return it
    }

    private fun all(ownerId: Int) = ctx.selectFrom(Tables.MODEL).where(Tables.MODEL.USER_ID.eq(ownerId))
    fun findById(ownerId: Int, modelId: Int?) = all(ownerId).and(Tables.MODEL.ID.eq(modelId)).fetchOne()
}