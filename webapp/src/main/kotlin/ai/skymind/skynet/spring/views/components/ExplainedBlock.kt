package ai.skymind.skynet.spring.views.components

import com.github.appreciated.card.Card
import com.vaadin.flow.component.Component
import com.vaadin.flow.component.html.Div
import com.vaadin.flow.component.orderedlayout.HorizontalLayout

class ExplainedBlock(explanation: Component, content: Component): Div() {

    init {
        val contentCard = Card(content).apply {
            width = "50%"
        }
        val explanationCard = Card(explanation).apply {
            width = "50%"
            addClassName("explained-block")
            addClassName("explanation")
        }

        add(HorizontalLayout(contentCard, explanationCard))
    }
}