package ai.skymind.skynet.spring.views

import ai.skymind.skynet.spring.views.layouts.LoginLayout
import ai.skymind.skynet.spring.views.state.UserSession
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.login.LoginOverlay
import com.vaadin.flow.component.orderedlayout.FlexComponent
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.router.Route

@Route(value ="", layout = LoginLayout::class)
class LoginView(
        val userSession: UserSession
) : VerticalLayout() {
    init {
        justifyContentMode = FlexComponent.JustifyContentMode.CENTER
        alignItems = FlexComponent.Alignment.CENTER

        height = "100%"


        val login = LoginOverlay().apply {
            setTitle("Pathmind")
            description = "Reinforcement Learning in the Cloud"
            isOpened = true
            addLoginListener {
                userSession.login(it.username, it.password)
                if(userSession.isLoggedIn()){
                    ui.ifPresent { it.navigate(ProjectListView::class.java) }
                    close()
                }else{
                    this.isError = true
                }
            }
        }

        add(login)
        val button = Button("Click me") { login.isOpened = true }
        add(button)
    }

}