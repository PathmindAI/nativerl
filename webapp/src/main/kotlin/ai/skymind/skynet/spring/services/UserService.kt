package ai.skymind.skynet.spring.services

import ai.skymind.skynet.data.db.jooq.Tables.USER
import org.jooq.DSLContext
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.stereotype.Service

@Service
class UserService(
        val ctx: DSLContext,
        val passwordEncoder: PasswordEncoder
){
    fun login(email: String, password: String): User? {
        val user = ctx.selectFrom(USER).where(USER.EMAIL.eq(email)).fetchOne()
        return user?.let{
            if(passwordEncoder.matches(password, it.password)){
                User(it.id, it.email)
            }else{
                null
            }
        }
    }
}

data class User(val id: Int, val username: String)