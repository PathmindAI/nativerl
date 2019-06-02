package ai.skymind.skynet.spring.views

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
import com.vaadin.flow.router.Route

@Route(value = "projects", layout = MainLayout::class)
class ProjectListView(
    val userSession: UserSession
) : VerticalLayout() {
    val grid = Grid(ProjectRecord::class.java)
    val filterText = TextField()

    init {
        grid.apply {
            setSelectionMode(Grid.SelectionMode.SINGLE)
            setColumns("name", "createdAt")
            addItemClickListener {
                ui.ifPresent { it.navigate(ExperimentListView::class.java) }
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
                H2("Projects"),
                Button("New Project").apply {
                    addClickListener { ui.ifPresent { it.navigate(ProjectCreateView::class.java) } }
                }
        ).apply {
            setWidthFull()
            alignItems = FlexComponent.Alignment.BASELINE
        })
        add(filterText)
        add(grid)

        updateList()
    }

    fun updateList() {
        val foundItems = userSession.findProject(filterText.value)
        grid.setItems(foundItems ?: emptyList())
    }
}
