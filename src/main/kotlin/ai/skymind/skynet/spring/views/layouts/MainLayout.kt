package ai.skymind.skynet.spring.views.layouts

import com.vaadin.flow.component.applayout.AppLayout
import com.vaadin.flow.component.dependency.HtmlImport
import com.vaadin.flow.component.html.Image
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.router.RouterLayout
import com.vaadin.flow.theme.Theme
import com.vaadin.flow.theme.lumo.Lumo

@Theme(Lumo::class)
@HtmlImport("frontend://styles/shared-styles.html")
class MainLayout: AppLayout(), RouterLayout {

    init {
        setBranding(VerticalLayout().apply {
            addClassName("topBar")
            add(Image("frontend/images/logo_skymind_white.svg", "Skymind Logo"))
        })

    }
}