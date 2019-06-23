package ai.skymind.skynet.spring.views

import ai.skymind.skynet.data.db.jooq.tables.records.ModelRecord
import ai.skymind.skynet.data.db.jooq.tables.records.ProjectRecord
import ai.skymind.skynet.spring.views.layouts.MainLayout
import ai.skymind.skynet.spring.views.state.UserSession
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.grid.Grid
import com.vaadin.flow.component.html.H2
import com.vaadin.flow.component.orderedlayout.FlexComponent
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.component.textfield.TextField
import com.vaadin.flow.data.value.ValueChangeMode
import com.vaadin.flow.router.BeforeEvent
import com.vaadin.flow.router.HasUrlParameter
import com.vaadin.flow.router.Route

@Route(value = "models", layout = MainLayout::class)
class ModelListView(
        val userSession: UserSession
) : VerticalLayout(), HasUrlParameter<Int> {
    var project: ProjectRecord? = null
    val grid = Grid(ModelRecord::class.java)
    val filterText = TextField()
    val title = H2("Models")

    init {
        grid.apply {
            setSelectionMode(Grid.SelectionMode.SINGLE)
            setColumns("name", "createdAt")
            addComponentColumn { model ->
                HorizontalLayout(
                        Button("Edit MDP") {
                            ui.ifPresent { it.navigate(EditMdpView::class.java, model.id) }
                        },
                        Button("Show runs") {
                            ui.ifPresent{ it.navigate(RunListView::class.java, model.id) }
                        }
                )
            }
        }

        filterText.apply {
            addClassName("align-right")

            placeholder = "search..."
            isClearButtonVisible = true
            valueChangeMode = ValueChangeMode.EAGER
            addValueChangeListener {
                updateList()
            }
        }

        add(HorizontalLayout(
                title,
                Button("Upload new model").apply {
                    addClickListener { ui.ifPresent { it.navigate(ExperimentCreateView::class.java) } }
                }
        ).apply {
            setWidthFull()
            alignItems = FlexComponent.Alignment.BASELINE
        })
        add(filterText)
        add(grid)
    }

    fun updateList() {
        val foundItems = userSession.findModels(project!!.id, filterText.value)
        grid.setItems(foundItems)
    }

    override fun setParameter(event: BeforeEvent?, projectId: Int?) {
        this.project = userSession.project(projectId)
        if(project != null) {
            title.text = "${project!!.name}: Models"
            updateList()
        }
    }
}