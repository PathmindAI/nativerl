package ai.skymind.skynet.spring.views

import ai.skymind.skynet.spring.views.layouts.MainLayout
import com.github.appreciated.card.Card
import com.juicy.JuicyAceEditor
import com.juicy.mode.JuicyAceMode
import com.juicy.theme.JuicyAceTheme
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.html.H2
import com.vaadin.flow.component.listbox.ListBox
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.component.splitlayout.SplitLayout
import com.vaadin.flow.router.Route
import org.springframework.core.io.ResourceLoader

@Route(value = "experiments/edit", layout = MainLayout::class)
class ExperimentEditView(
        val resourceLoader: ResourceLoader
) : VerticalLayout() {
    init {
        add(H2("ExperimentName"))
        add(HorizontalLayout(
                Button("Discard"),
                Button("Save & Close"),
                Button("Compile"),
                Button("Run Train Job")
        ))

        add(Card(SplitLayout(
                VerticalLayout(
                        ListBox<String>().apply {
                            setItems("Policy 1", "Policy 2", "Policy 3")
                            setWidthFull()
                        }
                ).apply {
                    addClassName("policy-list")
                    isPadding = false
                },
                JuicyAceEditor().apply {
                    value = resourceLoader.getResource("classpath:/files/mdp-template.java").inputStream.reader().readText()
                    setTheme(JuicyAceTheme.eclipse)
                    setMode(JuicyAceMode.java)
                    setWidthFull()
                    setHeightFull()
                }
        ).apply {
            setWidthFull()
            setHeightFull()
            setSplitterPosition(20.0)
        }).apply {
            setWidthFull()
            height = "80vh"
        })
    }
}