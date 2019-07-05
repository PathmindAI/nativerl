/*
 * This file is generated by jOOQ.
 */
package ai.skymind.skynet.data.db.jooq.tables;


import ai.skymind.skynet.data.db.jooq.Indexes;
import ai.skymind.skynet.data.db.jooq.Keys;
import ai.skymind.skynet.data.db.jooq.Public;
import ai.skymind.skynet.data.db.jooq.tables.records.MdpRecord;
import org.jooq.*;
import org.jooq.impl.DSL;
import org.jooq.impl.TableImpl;

import javax.annotation.Generated;
import java.sql.Timestamp;
import java.util.Arrays;
import java.util.List;


/**
 * This class is generated by jOOQ.
 */
@Generated(
    value = {
        "http://www.jooq.org",
        "jOOQ version:3.11.9"
    },
    comments = "This class is generated by jOOQ"
)
@SuppressWarnings({ "all", "unchecked", "rawtypes" })
public class Mdp extends TableImpl<MdpRecord> {

    private static final long serialVersionUID = 2005059808;

    /**
     * The reference instance of <code>public.mdp</code>
     */
    public static final Mdp MDP = new Mdp();

    /**
     * The class holding records for this type
     */
    @Override
    public Class<MdpRecord> getRecordType() {
        return MdpRecord.class;
    }

    /**
     * The column <code>public.mdp.id</code>.
     */
    public final TableField<MdpRecord, Integer> ID = createField("id", org.jooq.impl.SQLDataType.INTEGER.nullable(false).defaultValue(org.jooq.impl.DSL.field("nextval('mdp_id_seq'::regclass)", org.jooq.impl.SQLDataType.INTEGER)), this, "");

    /**
     * The column <code>public.mdp.user_id</code>.
     */
    public final TableField<MdpRecord, Integer> USER_ID = createField("user_id", org.jooq.impl.SQLDataType.INTEGER.nullable(false), this, "");

    /**
     * The column <code>public.mdp.model_id</code>.
     */
    public final TableField<MdpRecord, Integer> MODEL_ID = createField("model_id", org.jooq.impl.SQLDataType.INTEGER.nullable(false), this, "");

    /**
     * The column <code>public.mdp.name</code>.
     */
    public final TableField<MdpRecord, String> NAME = createField("name", org.jooq.impl.SQLDataType.VARCHAR.nullable(false), this, "");

    /**
     * The column <code>public.mdp.code</code>.
     */
    public final TableField<MdpRecord, String> CODE = createField("code", org.jooq.impl.SQLDataType.VARCHAR.nullable(false), this, "");

    /**
     * The column <code>public.mdp.verified</code>.
     */
    public final TableField<MdpRecord, Boolean> VERIFIED = createField("verified", org.jooq.impl.SQLDataType.BOOLEAN.nullable(false).defaultValue(org.jooq.impl.DSL.field("false", org.jooq.impl.SQLDataType.BOOLEAN)), this, "");

    /**
     * The column <code>public.mdp.created_at</code>.
     */
    public final TableField<MdpRecord, Timestamp> CREATED_AT = createField("created_at", org.jooq.impl.SQLDataType.TIMESTAMP.nullable(false).defaultValue(org.jooq.impl.DSL.field("now()", org.jooq.impl.SQLDataType.TIMESTAMP)), this, "");

    /**
     * The column <code>public.mdp.updated_at</code>.
     */
    public final TableField<MdpRecord, Timestamp> UPDATED_AT = createField("updated_at", org.jooq.impl.SQLDataType.TIMESTAMP.nullable(false).defaultValue(org.jooq.impl.DSL.field("now()", org.jooq.impl.SQLDataType.TIMESTAMP)), this, "");

    /**
     * Create a <code>public.mdp</code> table reference
     */
    public Mdp() {
        this(DSL.name("mdp"), null);
    }

    /**
     * Create an aliased <code>public.mdp</code> table reference
     */
    public Mdp(String alias) {
        this(DSL.name(alias), MDP);
    }

    /**
     * Create an aliased <code>public.mdp</code> table reference
     */
    public Mdp(Name alias) {
        this(alias, MDP);
    }

    private Mdp(Name alias, Table<MdpRecord> aliased) {
        this(alias, aliased, null);
    }

    private Mdp(Name alias, Table<MdpRecord> aliased, Field<?>[] parameters) {
        super(alias, null, aliased, parameters, DSL.comment(""));
    }

    public <O extends Record> Mdp(Table<O> child, ForeignKey<O, MdpRecord> key) {
        super(child, key, MDP);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Schema getSchema() {
        return Public.PUBLIC;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Index> getIndexes() {
        return Arrays.<Index>asList(Indexes.MDP_PKEY);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Identity<MdpRecord, Integer> getIdentity() {
        return Keys.IDENTITY_MDP;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public UniqueKey<MdpRecord> getPrimaryKey() {
        return Keys.MDP_PKEY;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<UniqueKey<MdpRecord>> getKeys() {
        return Arrays.<UniqueKey<MdpRecord>>asList(Keys.MDP_PKEY);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<ForeignKey<MdpRecord, ?>> getReferences() {
        return Arrays.<ForeignKey<MdpRecord, ?>>asList(Keys.MDP__MDP_OWNER, Keys.MDP__MDP_FOR_MODEL);
    }

    public User user() {
        return new User(this, Keys.MDP__MDP_OWNER);
    }

    public Model model() {
        return new Model(this, Keys.MDP__MDP_FOR_MODEL);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Mdp as(String alias) {
        return new Mdp(DSL.name(alias), this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Mdp as(Name alias) {
        return new Mdp(alias, this);
    }

    /**
     * Rename this table
     */
    @Override
    public Mdp rename(String name) {
        return new Mdp(DSL.name(name), null);
    }

    /**
     * Rename this table
     */
    @Override
    public Mdp rename(Name name) {
        return new Mdp(name, null);
    }
}