package ai.skymind.skynet.spring.views.state

import ai.skymind.skynet.data.db.jooq.Tables
import ai.skymind.skynet.spring.services.*
import com.vaadin.flow.spring.annotation.VaadinSessionScope
import org.springframework.stereotype.Component
import java.io.File

@Component
@VaadinSessionScope
class UserSession(
        val experimentService: ExperimentService,
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
    fun addProject(name: String, model: File) = withUser{
        val project = Tables.PROJECT.newRecord().apply {
            setName(name)
            userId = it.id
        }
        projectService.add(project)
    }

    fun experiments() = experimentService.findAll()
    fun findExperiments(query: String) = experimentService.find(query)
    fun addExperiment(experiment: Experiment) = experimentService.add(experiment)

    fun <T> withUser(f: (User) -> T): T? = user?.let(f)
}