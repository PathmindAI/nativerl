package ai.skymind.skynet.spring.views.layouts

import ai.skymind.skynet.spring.views.ProjectListView
import ai.skymind.skynet.spring.views.state.UserSession
import com.vaadin.flow.component.html.Div
import com.vaadin.flow.router.BeforeEnterEvent
import com.vaadin.flow.router.BeforeEnterObserver
import com.vaadin.flow.router.ParentLayout
import com.vaadin.flow.router.RouterLayout

@ParentLayout(ApplicationLayout::class)
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