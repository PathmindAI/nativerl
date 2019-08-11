package ai.skymind.skynet.spring.views

import ai.skymind.skynet.data.db.jooq.tables.records.MdpRecord
import ai.skymind.skynet.spring.cloud.job.local.Environment
import ai.skymind.skynet.spring.cloud.job.local.RLConfig
import ai.skymind.skynet.spring.services.ExecutionService
import ai.skymind.skynet.spring.views.components.ExplainedBlock
import ai.skymind.skynet.spring.views.layouts.MainLayout
import ai.skymind.skynet.spring.views.state.UserSession
import com.juicy.JuicyAceEditor
import com.juicy.mode.JuicyAceMode
import com.juicy.theme.JuicyAceTheme
import com.vaadin.flow.component.Component
import com.vaadin.flow.component.HasValue
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.formlayout.FormLayout
import com.vaadin.flow.component.html.H2
import com.vaadin.flow.component.html.H3
import com.vaadin.flow.component.html.Span
import com.vaadin.flow.component.notification.Notification
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.component.textfield.TextArea
import com.vaadin.flow.component.textfield.TextField
import com.vaadin.flow.router.BeforeEvent
import com.vaadin.flow.router.HasUrlParameter
import com.vaadin.flow.router.Route
import java.net.URI
import javax.tools.JavaFileObject
import javax.tools.SimpleJavaFileObject
import kotlin.properties.Delegates


@Route(value = "model/mdp/edit", layout = MainLayout::class)
class EditMdpView(
        val userSession: UserSession,
        val executionService: ExecutionService
) : VerticalLayout(), HasUrlParameter<Int> {
    val rewardEditor = JuicyAceEditor()
    val importsEditor = JuicyAceEditor()
    val variablesEditor = JuicyAceEditor()
    val resetEditor = JuicyAceEditor()
    val metricsEditor = JuicyAceEditor()
    val title = H2("ExperimentName")
    val observationSpace = TextField { e -> mdp?.let { it.observationSpaceSize = e.value.toInt() } }
    val actionSpace = TextField { e -> mdp?.let { it.actionSpaceSize = e.value.toInt() } }
    val epochs = TextField { e -> mdp?.let { it.epochs = e.value.toInt() } }
    val simulationStepsLength = TextField { e -> mdp?.let { it.simulationStepsLength = e.value.toInt() } }
    val batchSize = TextField { e -> mdp?.let { it.batchSize = e.value.toInt() } }
    val experienceReplaySteps = TextField { e -> mdp?.let { it.experienceReplaySteps = e.value.toInt() } }
    val stepsPerUpdate = TextField { e -> mdp?.let { it.stepsPerUpdate = e.value.toInt() } }
    val warmupSteps = TextField { e -> mdp?.let { it.warmupSteps = e.value.toInt() } }

    var mdp: MdpRecord? by Delegates.observable(null as MdpRecord?) { property, oldValue, newValue ->
        newValue?.let {
            // This only sets the editor value when the mdp record **itself** changes. Not when any one of its properties
            // does change
            title.text = userSession.model(it.modelId)?.name

            observationSpace.value = it.observationSpaceSize?.toString() ?: ""
            actionSpace.value = it.actionSpaceSize?.toString() ?: ""
            epochs.value = it.epochs?.toString() ?: ""
            simulationStepsLength.value  = it.simulationStepsLength?.toString() ?: ""
            batchSize.value = it.batchSize?.toString() ?: ""
            experienceReplaySteps.value = it.experienceReplaySteps?.toString() ?: ""
            stepsPerUpdate.value = it.stepsPerUpdate?.toString() ?: ""
            warmupSteps.value = it.warmupSteps?.toString() ?: ""

            rewardEditor.value = it.reward
            importsEditor.value = it.imports
            variablesEditor.value = it.variables
            resetEditor.value = it.reset
            metricsEditor.value = it.metrics
        }
    }

    init {
        add(title)
        add(HorizontalLayout(
                Button("Save").apply {
                    addClickListener {
                        mdp?.store()
                        Notification.show("Saved.")
                    }
                },
                Button("Run").apply {
                    addClickListener {
                        mdp?.let{
                            executionService.runMdp(it)
                            Notification.show("Training Started.")
                            ui.get().navigate(RunListView::class.java, it.modelId)
                        }
                    }
                }
        ))

        add(H2("Basic Options"))
        add(FormLayout().apply {
            addFormItem(observationSpace, "Observations Count")
            addFormItem(actionSpace, "Possible Actions Count")
        })

        val rewardDescription = "This is where you enter the code for your reward function. You have the following variables available: agent (your Main Agent), before and after. The variables before and after are the result of calling your getObservation function before and after your doAction function was called. You have to assign the reward to the reward variable. \n\nFor example: \n\n reward = before[0] - after[0];"
        add(createEditor("Reward Function", rewardDescription, rewardEditor, RLConfig.CompileErrors::reward) { e, mdp -> mdp.reward = e.value })

        add(H2("Advanced Options"))
        add(FormLayout().apply {
            addFormItem(epochs, "Epochs")
            addFormItem(simulationStepsLength, "Simulation Steps Count")
            addFormItem(batchSize, "Batch Size")
            addFormItem(experienceReplaySteps, "Experience Replay Steps")
            addFormItem(stepsPerUpdate, "Steps per Update")
            addFormItem(warmupSteps, "Warm up Steps")
        })



        val importsDescription = "If you need any additional classes in the functions you define here, you can import them here"
        add(createEditor("Imports", importsDescription, importsEditor, RLConfig.CompileErrors::imports) { e, mdp -> mdp.imports = e.value })

        val variablesDescription = "If you need additional variables that are going to be available in all of the functions defined here, you can add them in this field."
        add(createEditor("Class Variables", variablesDescription, variablesEditor, RLConfig.CompileErrors::variables) { e, mdp -> mdp.variables = e.value })

        val resetDescription = "If you need to do any additional setup before the simulation can be used, you can do it with this function. You have to following variables available: agent (your Main Agent)."
        add(createEditor("Reset Function", resetDescription, resetEditor, RLConfig.CompileErrors::reset) { e, mdp -> mdp.reset = e.value })

        val metricsDescription = "If you want to collect any additional metrics during the training, you can do so with this function. You have to following variables available: agent (your Main Agent)."
        add(createEditor("Metrics Function", metricsDescription, metricsEditor, RLConfig.CompileErrors::metrics) { e, mdp -> mdp.metrics = e.value })
    }

    private fun createEditor(title: String, explanation: String, editor: JuicyAceEditor, errorProperty: (RLConfig.CompileErrors) -> MutableList<String>, onChange: (e: HasValue.ValueChangeEvent<String>, mdp: MdpRecord) -> Unit): Component {
        val compileOutput = TextArea().apply {
            isReadOnly = true
            setWidthFull()
            isVisible = false
        }

        return VerticalLayout(
                H3(title),
                ExplainedBlock(
                        Span(explanation),
                        editor.apply {
                            value = ""
                            setTheme(JuicyAceTheme.eclipse)
                            setMode(JuicyAceMode.java)
                            setWidthFull()
                            height = "15em"
                            addValueChangeListener { e ->
                                mdp?.let {
                                    onChange(e, it)
                                    val allErrors = RLConfig("output", Environment(emptyList()), userSession.model(it.modelId)!!,it).compileErrors()
                                    val errors = errorProperty(allErrors)
                                    if(errors.size > 0){
                                        compileOutput.value = errors.joinToString("\n")
                                        compileOutput.isVisible = true
                                    }else{
                                        compileOutput.value = ""
                                        compileOutput.isVisible  = false
                                    }
                                }
                            }
                        }
                ).apply{
                    setWidthFull()
                },
                compileOutput
        )
    }

    override fun setParameter(event: BeforeEvent?, modelId: Int?) {
        val mdps = userSession.findMdps(modelId!!)
        mdp = mdps?.firstOrNull() ?: userSession.newMdp(modelId)
    }


}

internal class CharSequenceJavaFileObject(className: String, val content: CharSequence) : SimpleJavaFileObject(URI.create("string:///" + className.replace('.', '/') + JavaFileObject.Kind.SOURCE.extension), JavaFileObject.Kind.SOURCE) {
    override fun getCharContent(ignoreEncodingErrors: Boolean): CharSequence {
        return content
    }
}