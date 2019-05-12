package ai.skymind.skynet.spring.views

import ai.skymind.skynet.spring.services.Experiment
import ai.skymind.skynet.spring.services.ExperimentService
import com.vaadin.flow.component.applayout.AppLayout
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.grid.Grid
import com.vaadin.flow.component.html.H2
import com.vaadin.flow.component.html.Span
import com.vaadin.flow.component.notification.Notification
import com.vaadin.flow.component.orderedlayout.FlexComponent
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.component.textfield.TextField
import com.vaadin.flow.data.value.ValueChangeMode
import com.vaadin.flow.router.Route

@Route(value = "experiments")
class ExperimentListView(
        val experimentService: ExperimentService
): AppLayout() {
    val grid = Grid(Experiment::class.java)
    val filterText = TextField()

    init {
        setBranding(Span("Skymind"))
        setContent(VerticalLayout().apply {
            grid.apply {
                setSelectionMode(Grid.SelectionMode.SINGLE)
                setColumns("name", "dateCreated", "runs")
                addComponentColumn { experiment ->
                    HorizontalLayout(
                            Button("Run") {
                                Notification.show("Running ${experiment.name}")
                            },
                            Button("Policies") {
                                ui.ifPresent { it.navigate(ExperimentEditView::class.java) }
                            }
                    )
                }
            }

            filterText.apply {
                placeholder = "search..."
                isClearButtonVisible = true
                valueChangeMode = ValueChangeMode.EAGER
                addValueChangeListener {
                    updateList()
                }
            }

            add(HorizontalLayout(
                    H2("Experiments"),
                    Button("New Experiment").apply{
                        addClickListener { ui.ifPresent { it.navigate(ExperimentCreateView::class.java) } }
                    }
            ).apply {
                setWidthFull()
                alignItems = FlexComponent.Alignment.BASELINE
            })
            add(filterText)
            add(grid)

            updateList()
        })
    }

    fun updateList(){
        val foundItems = experimentService.find(filterText.value)
        grid.setItems(foundItems)
    }
}