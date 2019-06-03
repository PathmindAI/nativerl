try {
    //record the training data in rl4j-data in a new folder (save)
    DataManager manager = new DataManager(true);

    QLearning.QLConfiguration AL_QL =
            new QLearning.QLConfiguration(
                    1,    //Random seed
                    2880,    //Max step By epoch (episode ends @ (this * time step) time units
                    288000, //Max step
                    288000, //Max size of experience replay
                    128,    //size of batches
                    500,    //target update (hard)
                    10,     //num step noop warmup
                    0.1,    //reward scaling
                    0.99,   //gamma
                    1.0,    //td-error clipping
                    0.1f,   //min epsilon
                    1000,   //num step for eps greedy anneal
                    true    //double DQN
            );

    DQNFactoryStdDense.Configuration AL_NET =
            DQNFactoryStdDense.Configuration.builder()
                    .l2(0).updater(new RmsProp(0.001)).numHiddenNodes(300).numLayer(2).build();

    //define the training
    QLearningDiscreteDense<Encodable> dql = new QLearningDiscreteDense(mdp, AL_NET, AL_QL, manager);

    //train
    dql.train();

    //get the final policy
    DQNPolicy<Encodable> pol = dql.getPolicy();

    //serialize and save (serialization showcase, but not required)
    // Ensure a previous policy doesn't get overwritten.
    String fn = "PhasePolicy";
    String ext = ".zip";
    int n = 0;
    File outfile = new File(fn + ext);
    while (outfile.exists())
        outfile = new File(fn + "_" + (n++) + ext);
    if (n == 0)
        pol.save(fn + ext);
    else
        pol.save(fn + "_" + n + ext);

    //close the mdp (close engine)
    mdp.close();

} catch (IOException e) {
    e.printStackTrace();
}