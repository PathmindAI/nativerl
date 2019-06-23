package ai.skymind.skynet.spring.db

import ai.skymind.skynet.data.db.jooq.Tables
import ai.skymind.skynet.data.db.jooq.tables.records.RunRecord
import org.jooq.DSLContext
import org.springframework.stereotype.Service

@Service
class RunRepository(
        val ctx: DSLContext
) {
    fun add(it: RunRecord): RunRecord {
        it.attach(ctx.configuration())
        it.store()
        return it
    }

    private fun all(ownerId: Int) = ctx.selectFrom(Tables.RUN).where(Tables.RUN.USER_ID.eq(ownerId))

    fun findByModelId(ownerId: Int, modelId: Int, query: String?): List<RunRecord> = all(ownerId).and(Tables.RUN.MODEL_ID.eq(modelId)).let {
        when(query){
            "", null -> it
            else -> it.and(Tables.RUN.EXTERNAL_JOB_ID.containsIgnoreCase(query))
        }
    }.fetch().toList()
}