package ai.skymind.skynet.spring.views.layouts

import com.vaadin.flow.component.dependency.HtmlImport
import com.vaadin.flow.component.html.Div
import com.vaadin.flow.component.page.Push
import com.vaadin.flow.router.RouterLayout
import com.vaadin.flow.theme.Theme
import com.vaadin.flow.theme.lumo.Lumo

@Push
@Theme(Lumo::class)
@HtmlImport("frontend://styles/shared-styles.html")
class ApplicationLayout(): Div(), RouterLayout