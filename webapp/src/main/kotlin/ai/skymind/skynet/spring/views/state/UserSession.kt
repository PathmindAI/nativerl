package ai.skymind.skynet.spring.views.state

import ai.skymind.skynet.data.db.jooq.tables.records.ModelRecord
import ai.skymind.skynet.spring.db.ModelRepository
import ai.skymind.skynet.spring.services.ProjectService
import ai.skymind.skynet.spring.services.User
import ai.skymind.skynet.spring.services.UserService
import com.vaadin.flow.spring.annotation.VaadinSessionScope
import org.springframework.stereotype.Component
import java.io.File

@Component
@VaadinSessionScope
class UserSession(
        val modelService: ModelRepository,
        val projectService: ProjectService,
        val userService: UserService
) {
    var user: User? = null

    fun login(username: String, password: String){
        user = userService.login(username, password)
    }

    fun isLoggedIn() = user != null

    fun canAccess(view: Class<*>): Boolean = when(user){
        null -> false
        else -> true
    }

    fun projects() = withUser { projectService.findAll(it.id) }
    fun findProject(query: String) = withUser { projectService.find(it.id, query) }
    fun addProject(name: String, model: File) = withUser{ projectService.addProject(it.id, name, model) }

    fun experiments(): Any = TODO()
    fun findExperiments(query: String): Any = TODO()
    fun addExperiment(experiment: Any): Any = TODO()

    fun <T> withUser(f: (User) -> T): T? = user?.let(f)
    fun findModels(projectId: Int?, query: String?): List<ModelRecord> = when(projectId) {
        null -> emptyList()
        else -> when(query) {
            null -> emptyList()
            else -> projectService.models(projectId).filter { it.name.contains(query, true) }
        }
    }
}