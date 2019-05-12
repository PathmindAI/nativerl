package ai.skymind.skynet.spring.views

import ai.skymind.skynet.spring.services.Project
import ai.skymind.skynet.spring.services.ProjectService
import com.vaadin.flow.component.applayout.AppLayout
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.formlayout.FormLayout
import com.vaadin.flow.component.html.H2
import com.vaadin.flow.component.html.Span
import com.vaadin.flow.component.orderedlayout.FlexComponent
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.component.textfield.TextField
import com.vaadin.flow.component.upload.Upload
import com.vaadin.flow.component.upload.receivers.FileBuffer
import com.vaadin.flow.router.Route

@Route("projects/create")
class ProjectCreateView(
        val projectService: ProjectService
): AppLayout() {
    val projectName = TextField()
    init {
        setBranding(Span("Skymind"))
        setContent(VerticalLayout().apply{
            add(H2("Create Project"))
            add(FormLayout().apply {
                addFormItem(projectName, "Project Name")
            })
            add(Upload(FileBuffer()).apply {
                dropLabel = Span("Drag exported Model as Zip File here")
            })
            add(HorizontalLayout(
                    Button("Cancel").apply{
                        addClickListener { ui.ifPresent { it.navigate(ProjectListView::class.java) } }
                    },
                    Button("Create Project").apply {
                        addClickListener {
                            projectService.add(Project(projectName.value))
                            ui.ifPresent { it.navigate(ProjectListView::class.java) }
                        }
                    }
            ).apply {
                setWidthFull()
                justifyContentMode = FlexComponent.JustifyContentMode.END
            })
        })
    }
}