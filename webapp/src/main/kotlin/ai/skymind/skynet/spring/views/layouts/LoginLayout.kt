package ai.skymind.skynet.spring.views.layouts

import ai.skymind.skynet.spring.views.ProjectListView
import ai.skymind.skynet.spring.views.state.UserSession
import com.vaadin.flow.component.dependency.HtmlImport
import com.vaadin.flow.component.html.Div
import com.vaadin.flow.router.BeforeEnterEvent
import com.vaadin.flow.router.BeforeEnterObserver
import com.vaadin.flow.router.RouterLayout
import com.vaadin.flow.theme.Theme
import com.vaadin.flow.theme.lumo.Lumo

@Theme(Lumo::class)
@HtmlImport("frontend://styles/shared-styles.html")
class LoginLayout(
        val userSession: UserSession
): Div(), RouterLayout, BeforeEnterObserver {
    init {
    }

    override fun beforeEnter(e: BeforeEnterEvent){
        if(userSession.isLoggedIn()){
            e.forwardTo(ProjectListView::class.java)
        }
    }
}