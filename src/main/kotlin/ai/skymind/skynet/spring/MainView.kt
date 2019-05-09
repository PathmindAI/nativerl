package ai.skymind.skynet.spring

import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.notification.Notification
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.router.Route
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.stereotype.Service
import java.time.LocalTime

@Route
class MainView(@Autowired bean: MessageBean) : VerticalLayout() {

    init {
        val button = Button("Click me") { e -> Notification.show(bean.message) }
        add(button)
    }

}


@Service
class MessageBean {

    val message: String
        get() = "Button was clicked at " + LocalTime.now()
}
