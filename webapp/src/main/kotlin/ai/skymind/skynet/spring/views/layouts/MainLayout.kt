package ai.skymind.skynet.spring.views.layouts

import ai.skymind.skynet.spring.views.LoginView
import ai.skymind.skynet.spring.views.state.UserSession
import com.vaadin.flow.component.applayout.AppLayout
import com.vaadin.flow.component.dependency.HtmlImport
import com.vaadin.flow.component.html.Div
import com.vaadin.flow.component.html.Image
import com.vaadin.flow.component.html.Span
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.router.BeforeEnterEvent
import com.vaadin.flow.router.BeforeEnterObserver
import com.vaadin.flow.router.RouterLayout
import com.vaadin.flow.theme.Theme
import com.vaadin.flow.theme.lumo.Lumo

@Theme(Lumo::class)
@HtmlImport("frontend://styles/shared-styles.html")
class MainLayout(
        val userSession: UserSession
): AppLayout(), RouterLayout, BeforeEnterObserver {

    init {
        setBranding(HorizontalLayout().apply {
            addClassName("topBranding")
            setWidthFull()

            add(Image("frontend/images/logo_skymind_white.svg", "Skymind Logo"))
            add(Div().apply {
                addClassName("expand")
            })
        })

        setMenu(HorizontalLayout().apply {
            setWidthFull()
            addClassName("topMenu")
            if(userSession.isLoggedIn()){
                add(Span(userSession.user!!.username))
            }
        })
    }

    override fun beforeEnter(e: BeforeEnterEvent){
        if(!userSession.canAccess(e.navigationTarget)){
            e.forwardTo(LoginView::class.java)
        }
    }

}