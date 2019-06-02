package ai.skymind.skynet.spring.views.state

import ai.skymind.skynet.spring.services.*
import com.vaadin.flow.spring.annotation.VaadinSessionScope
import org.springframework.stereotype.Component

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

    fun projects() = projectService.findAll()
    fun findProject(query: String) = projectService.find(query)
    fun addProject(project: Project) = projectService.add(project.copy(owner=user!!.username))

    fun experiments() = experimentService.findAll()
    fun findExperiments(query: String) = experimentService.find(query)
    fun addExperiment(experiment: Experiment) = experimentService.add(experiment)
}