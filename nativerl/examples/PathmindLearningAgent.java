package com.pathmind.anylogic;

import com.anylogic.engine.LearningAgentInterface;
import com.anylogic.rl.data.Action;
import com.anylogic.rl.data.Configuration;
import com.anylogic.rl.data.Observation;

public class PathmindLearningAgent<Obs extends Observation, Act extends Action, Config extends Configuration> implements LearningAgentInterface<Obs,Act,Config> {

	@SuppressWarnings("unchecked")
	public Act takeAction(Obs obs, Act act) {
		// get obs extends Observaion via  reflection

		// translate it to double[]

		// call this.computeActions(double[])

		// convert float[] to act extends Action

		// return act

		throw new RuntimeException("takeAction is not allowed to run for now");
	}
}
