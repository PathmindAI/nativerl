package ai.skymind.skynet.spring.views

import ai.skymind.skynet.data.db.jooq.tables.records.RunRecord
import ai.skymind.skynet.spring.cloud.job.api.JobExecutor
import ai.skymind.skynet.spring.views.layouts.MainLayout
import ai.skymind.skynet.spring.views.state.UserSession
import com.vaadin.flow.component.AttachEvent
import com.vaadin.flow.component.DetachEvent
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.dialog.Dialog
import com.vaadin.flow.component.grid.Grid
import com.vaadin.flow.component.html.Anchor
import com.vaadin.flow.component.html.H2
import com.vaadin.flow.component.orderedlayout.FlexComponent
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.component.textfield.TextArea
import com.vaadin.flow.component.textfield.TextField
import com.vaadin.flow.data.value.ValueChangeMode
import com.vaadin.flow.router.BeforeEvent
import com.vaadin.flow.router.HasUrlParameter
import com.vaadin.flow.router.Route
import com.vaadin.flow.server.InputStreamFactory
import com.vaadin.flow.server.StreamResource
import org.slf4j.LoggerFactory

@Route(value = "modelRuns", layout = MainLayout::class)
class RunListView(
    val userSession: UserSession,
    val jobExecutor: JobExecutor
) : VerticalLayout(), HasUrlParameter<Int> {
    var modelId: Int? = null
    val grid = Grid(RunRecord::class.java)
    val filterText = TextField()

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
        backgroundThread!!.name = "Update Run List"
        backgroundThread!!.start()
    }

    override fun onDetach(detachEvent: DetachEvent?) {
        backgroundThread?.interrupt()
    }

    init {
        grid.apply {
            setSelectionMode(Grid.SelectionMode.SINGLE)
            setColumns("externalJobId", "startedAt", "status")
            addComponentColumn { run ->
                val resource = StreamResource("Policy.zip", InputStreamFactory {
                    jobExecutor.getPolicy(run.externalJobId)
                })
                HorizontalLayout().apply {
                    if(!listOf("Completed", "User terminated", "Stopping").contains(run.status)){
                        add(Button("Stop"){
                            jobExecutor.stop(run.externalJobId)
                            updateList()
                        }.apply {
                            isDisableOnClick = true
                        })
                    }
                    if(listOf("Completed", "Started", "Executing", "Failed").contains(run.status)){
                        add(Button("Console") {
                            var open = true
                            val console = TextArea().apply {
                                setWidthFull()
                                setHeightFull()
                                isReadOnly = true
                            }


                            var listener: Thread? = null
                            if(run.status == "Executing") {
                                listener = Thread {
                                    while(open){
                                        try {
                                            val tailConsole = jobExecutor.tailConsoleOutput(run.externalJobId)
                                            access {
                                                console.value = tailConsole
                                            }
                                        }catch (e: Exception){
                                            logger.error(e.message, e)
                                        }
                                        Thread.sleep(1000)
                                    }
                                }
                                listener.name = "Update Console Output"
                                listener.start()
                            }else{
                                console.value = jobExecutor.getConsoleOutput(run.externalJobId)
                            }
                            Dialog().apply {
                                add(console)
                                height = "80vh"
                                width = "80vw"
                                open()
                                addDialogCloseActionListener {
                                    open = false
                                    listener?.interrupt()
                                    close()
                                }
                                addDetachListener {
                                    open = false
                                    listener?.interrupt()
                                }
                            }
                        })
                    }
                    if(run.status == "Completed"){
                        add(Anchor().apply {
                            add(Button("Download Policy"))
                            element.apply {
                                setAttribute("href", resource)
                                setAttribute("download", true)
                            }
                        })
                    }
                }
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
                H2("Model Training Runs")
        ).apply {
            setWidthFull()
            alignItems = FlexComponent.Alignment.BASELINE
        })
        add(filterText)
        add(grid)
    }

    fun updateList() {
        val foundItems = userSession.findRuns(modelId!!, filterText.value)
        foundItems?.filter {
            !listOf("Completed", "User terminated").contains(it.status)
        }?.forEach {
            it.status = jobExecutor.status(it.externalJobId)
            it.store()
        }
        grid.setItems(foundItems ?: emptyList())
    }

    override fun setParameter(event: BeforeEvent?, parameter: Int?) {
        modelId = parameter
        updateList()
    }

    private fun access(function: () -> Unit) {
        ui.ifPresent{
            it.access(function)
        }
    }


}
