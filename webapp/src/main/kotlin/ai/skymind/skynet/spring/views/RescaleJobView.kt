package ai.skymind.skynet.spring.views

import ai.skymind.skynet.spring.cloud.job.rescale.rest.RescaleRestApiClient
import ai.skymind.skynet.spring.cloud.job.rescale.rest.entities.JobSummary
import ai.skymind.skynet.spring.views.layouts.MainLayout
import ai.skymind.skynet.spring.views.state.UserSession
import com.vaadin.flow.component.AttachEvent
import com.vaadin.flow.component.DetachEvent
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.dialog.Dialog
import com.vaadin.flow.component.grid.Grid
import com.vaadin.flow.component.html.H2
import com.vaadin.flow.component.orderedlayout.FlexComponent
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.component.textfield.TextArea
import com.vaadin.flow.router.Route
import org.slf4j.LoggerFactory

/**
 * Internal View used to check on the current Rescale Job Status.
 *
 * TODO: For now this is security by obscurity. It only requires that any valid user is logged in.
 */
@Route(value = "rescale", layout = MainLayout::class)
class RescaleJobView(
    val userSession: UserSession,
    val rescaleRestApiClient: RescaleRestApiClient
) : VerticalLayout() {
    val logger = LoggerFactory.getLogger(RescaleJobView::class.java)
    var backgroundThread: Thread? = null

    override fun onAttach(attachEvent: AttachEvent?) {
        backgroundThread = Thread{
            while(true){
                    ui.get().access{
                        try {
                            updateList()
                        }catch(e: Exception) {
                            println(e)
                        }
                    }
                    Thread.sleep(10000)
            }
        }
        backgroundThread!!.start()
    }

    override fun onDetach(detachEvent: DetachEvent?) {
        backgroundThread?.interrupt()
    }

    val grid = Grid(JobSummary::class.java)

    init {
        setHeightFull()

        grid.apply {
            setHeightFull()
            setSelectionMode(Grid.SelectionMode.SINGLE)
            setColumns("id", "name")
            addColumn {it.dateInserted.toLocalDate()}.setHeader("Date")
            addColumn { it.dateInserted.toLocalTime() }.setHeader("Time")
            addColumn { it.jobStatus.content }.setHeader ("Job Status" )
            addColumn { it.clusterStatusDisplay?.content ?: "N/A" }.setHeader("Cluster Status")
            addComponentColumn { summary ->
                HorizontalLayout().apply {
                    if(summary.jobStatus.content != "Completed" && summary.clusterStatusDisplay?.content != "Stopped"){
                        add(Button("Stop").apply {
                            addClickListener {
                                rescaleRestApiClient.jobStop(summary.id)
                                updateList()
                            }
                        })
                    }
                    if(listOf("Started", "Stopped").contains(summary.clusterStatusDisplay?.content)){
                        add(Button("Console") {
                            var open = true
                            val console = TextArea().apply {
                                setWidthFull()
                                setHeightFull()
                                isReadOnly = true
                            }


                            if(summary.jobStatus.content != "Completed" && summary.clusterStatusDisplay?.content != "Stopped") {
                                Thread {
                                    while(open){
                                        try {
                                            val tailConsole = rescaleRestApiClient.tailConsole(summary.id, "1")
                                           access {
                                                    console.value = tailConsole
                                                }
                                        }catch (e: Exception){
                                            logger.error(e.message, e)
                                        }
                                        Thread.sleep(1000)
                                    }
                                }.start()
                            }else{
                                console.value = rescaleRestApiClient.consoleOutput(summary.id) + "\n" + rescaleRestApiClient.compileOutput(summary.id)
                            }
                            Dialog().apply {
                                add(console)
                                height = "80vh"
                                width = "80vw"
                                open()
                                addDialogCloseActionListener {
                                    open = false
                                    close()
                                }
                            }
                        })
                    }
                }
            }
        }

        add(HorizontalLayout(
                H2("Rescale Jobs")
        ).apply {
            setWidthFull()
            alignItems = FlexComponent.Alignment.BASELINE
        })
        add(grid)
    }

    private fun access(function: () -> Unit) {
        ui.ifPresent{
            it.access(function)
        }
    }


    fun updateList() {
        val foundItems = rescaleRestApiClient.jobList()
        grid.setItems(foundItems.results)
    }


}