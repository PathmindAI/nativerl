package ai.skymind.skynet.spring.db

import ai.skymind.skynet.data.db.jooq.Tables
import ai.skymind.skynet.data.db.jooq.tables.records.MdpRecord
import org.jooq.DSLContext
import org.springframework.core.io.ResourceLoader
import org.springframework.stereotype.Service

@Service
class MdpRepository(
        val ctx: DSLContext,
        val resourceLoader: ResourceLoader
        ) {
    fun findAllByModelId(ownerId: Int, modelId: Int) = all(ownerId).and(Tables.MDP.MODEL_ID.eq(modelId)).fetch().toList()
    fun add(it: MdpRecord): MdpRecord {
        it.attach(ctx.configuration())
        it.store()
        return it
    }

    private fun all(ownerId: Int) = ctx.selectFrom(Tables.MDP).where(Tables.MDP.USER_ID.eq(ownerId))
    fun findById(ownerId: Int, mdpId: Int?) = all(ownerId).and(Tables.MDP.ID.eq(mdpId)).fetchOne()
    fun newMdp(ownerId: Int, modelId: Int): MdpRecord {
        val mdp = Tables.MDP.newRecord().apply {
            name = "Initial MDP"
            userId = ownerId
            setModelId(modelId)
            code = resourceLoader.getResource("classpath:/files/mdp-template.java").inputStream.reader().readText()
        }

        mdp.attach(ctx.configuration())
        mdp.insert()
        return mdp
    }
}