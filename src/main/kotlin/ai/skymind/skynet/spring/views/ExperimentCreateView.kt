package ai.skymind.skynet.spring.views

import ai.skymind.skynet.spring.services.Experiment
import ai.skymind.skynet.spring.views.layouts.MainLayout
import ai.skymind.skynet.spring.views.state.UserSession
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.formlayout.FormLayout
import com.vaadin.flow.component.html.H2
import com.vaadin.flow.component.orderedlayout.FlexComponent
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.component.textfield.TextField
import com.vaadin.flow.router.Route

@Route("experiments/create", layout = MainLayout::class)
class ExperimentCreateView(
    val userSession: UserSession
) : VerticalLayout() {
    val experimentName = TextField()

    init {
        add(H2("Create Experiment"))
        add(FormLayout().apply {
            addFormItem(experimentName, "Experiment Name")
        })
        add(HorizontalLayout(
                Button("Cancel").apply {
                    addClickListener { ui.ifPresent { it.navigate(ExperimentListView::class.java) } }
                },
                Button("Create Experiment").apply {
                    addClickListener {
                        userSession.addExperiment(Experiment(experimentName.value))
                        ui.ifPresent { it.navigate(ExperimentListView::class.java) }
                    }
                }
        ).apply {
            setWidthFull()
            justifyContentMode = FlexComponent.JustifyContentMode.END
        })

    }
}