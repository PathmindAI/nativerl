package ai.skymind.skynet.spring.views

import ai.skymind.skynet.data.db.jooq.tables.records.MdpRecord
import ai.skymind.skynet.spring.services.ExecutionService
import ai.skymind.skynet.spring.views.layouts.MainLayout
import ai.skymind.skynet.spring.views.state.UserSession
import com.github.appreciated.card.Card
import com.juicy.JuicyAceEditor
import com.juicy.mode.JuicyAceMode
import com.juicy.theme.JuicyAceTheme
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.html.H2
import com.vaadin.flow.component.notification.Notification
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.router.BeforeEvent
import com.vaadin.flow.router.HasUrlParameter
import com.vaadin.flow.router.Route
import kotlin.properties.Delegates

@Route(value = "model/mdp/edit", layout = MainLayout::class)
class EditMdpView(
        val userSession: UserSession,
        val executionService: ExecutionService
) : VerticalLayout(), HasUrlParameter<Int> {
    var editor = JuicyAceEditor()
    var mdp: MdpRecord? by Delegates.observable(null as MdpRecord?) { property, oldValue, newValue ->
        newValue?.let {
            editor.value = it.code
        }
    }

    init {
        add(H2("ExperimentName"))
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
                        }
                    }
                }
        ))

        add(Card(
                editor.apply {
                    value = ""
                    setTheme(JuicyAceTheme.eclipse)
                    setMode(JuicyAceMode.java)
                    setWidthFull()
                    setHeightFull()
                    addValueChangeListener { e ->
                        mdp?.let {
                            it.code = e.value
                        }
                    }
                }
        ).apply {
            setWidthFull()
            height = "80vh"
        })
    }

    override fun setParameter(event: BeforeEvent?, modelId: Int?) {
        val mdps = userSession.findMdps(modelId!!)
        mdp = mdps?.firstOrNull() ?: userSession.newMdp(modelId)
    }

}