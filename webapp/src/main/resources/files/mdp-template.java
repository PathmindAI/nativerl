new MDP<Encodable, Integer, DiscreteSpace>() {
    Engine engine;
    Main root;
    int simCount = 0;
    String combinations[][] = {
            {"constant_moderate", "constant_moderate"},
            {"none_til_heavy_afternoon_peak", "constant_moderate"},
            {"constant_moderate", "none_til_heavy_afternoon_peak"},
            {"peak_afternoon", "peak_morning"},
            {"peak_morning", "peak_afternoon"}
    };

    ObservationSpace<Encodable> observationSpace = new ArrayObservationSpace<>(new int[]{10});
    DiscreteSpace actionSpace = new DiscreteSpace(2);

    public ObservationSpace<Encodable> getObservationSpace() {
        return observationSpace;
    }

    public DiscreteSpace getActionSpace() {
        return actionSpace;
    }

    public Encodable getObservation() {
        return new Encodable() {
            double[] a = root.getState();

            public double[] toArray() {
                return a;
            }
        };
    }

    public Encodable reset() {
        simCount++;
        if (engine != null) {
            engine.stop();
        }
        // Create Engine, initialize random number generator:
        engine = createEngine();
        // Fixed seed (reproducible simulation runs)
        engine.getDefaultRandomGenerator().setSeed(1);
        // Selection mode for simultaneous events:
        engine.setSimultaneousEventsSelectionMode(Engine.EVENT_SELECTION_LIFO);
        // Set stop time:
        engine.setStopTime(28800);
        // Create new root object:
        root = new Main(engine, null, null);
        root.policy = null;
        // TODO Setup parameters of root object here
        root.setParametersToDefaultValues();
        root.usePolicy = false;
        root.manual = false;
        root.schedNameNS = combinations[simCount % combinations.length][0];
        root.schedNameEW = combinations[simCount % combinations.length][1];

        // Prepare Engine for simulation:
        engine.start(root);


        return getObservation();
    }

    public void close() {
        // Destroy the model:
        engine.stop();
    }

    public StepReply<Encodable> step(Integer action) {
        double[] s0 = root.getState();
        int p0 = root.trafficLight.getCurrentPhaseIndex();
        // {modified step function to ignore action if in yellow phase}
        root.step(action);
        engine.runFast(root.time() + 10);
        double[] s1 = root.getState();
        int p1 = root.trafficLight.getCurrentPhaseIndex();

        double sum0 = s0[0] + s0[2] + s0[4] + s0[6];
        double sum1 = s1[0] + s1[2] + s1[4] + s1[6];
        double reward = sum0 < sum1 ? -1 : sum0 > sum1 ? 1 : 0;

        return new StepReply(getObservation(), reward, isDone(), null);
    }

    public boolean isDone() {
        return false;
    }

    public MDP<Encodable, Integer, DiscreteSpace> newInstance() {
        return null; // not required for DQN
    }
};