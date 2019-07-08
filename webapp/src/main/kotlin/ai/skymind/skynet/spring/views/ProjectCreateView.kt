package ai.skymind.skynet.spring.views

import ai.skymind.skynet.spring.views.layouts.MainLayout
import ai.skymind.skynet.spring.views.state.UserSession
import ai.skymind.skynet.spring.views.upload.MultiFileBuffer
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.formlayout.FormLayout
import com.vaadin.flow.component.html.H2
import com.vaadin.flow.component.html.Span
import com.vaadin.flow.component.notification.Notification
import com.vaadin.flow.component.orderedlayout.FlexComponent
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.component.textfield.TextField
import com.vaadin.flow.component.upload.Upload
import com.vaadin.flow.router.Route
import java.nio.file.Files

@Route("projects/create", layout = MainLayout::class)
class ProjectCreateView(
        val userSession: UserSession
) : VerticalLayout() {
    val projectName = TextField()
    val fileBuffer = MultiFileBuffer()

    init {
        add(H2("Create Project"))
        add(FormLayout().apply {
            addFormItem(projectName, "Project Name")
        })
        add(Upload(fileBuffer).apply {
            dropLabel = Span("Drag exported Model as Zip File here")
        })
        add(HorizontalLayout(
                Button("Cancel").apply {
                    addClickListener { ui.ifPresent { it.navigate(ProjectListView::class.java) } }
                },
                Button("Create Project").apply {
                    isDisableOnClick = true
                    addClickListener {
                        Notification.show("Uploading Model...")
                        fileBuffer.getFiles().forEach{
                            val path = Files.createTempDirectory("pathmind-upload")
                            val file = path.resolve("model.zip").toFile()
                            val target = file.outputStream()
                            fileBuffer.getInputStream(it).copyTo(target)
                            target.close()
                            userSession.addProject(projectName.value, file)
                            path.toFile().deleteRecursively()
                        }
                        ui.ifPresent { it.navigate(ProjectListView::class.java) }
                    }
                }
        ).apply {
            setWidthFull()
            justifyContentMode = FlexComponent.JustifyContentMode.END
        })
    }
}