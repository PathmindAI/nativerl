grep -m 1 "python3: can't open file 'rllibtrain.py'" process_output.log >> errors.log ;
grep -m 1 "SyntaxError: invalid syntax" process_output.log >> errors.log ;
grep -m 1 "Fatal Python error: Segmentation fault" process_output.log >> errors.log ;
grep -m 1 "Worker crashed during call to train()" process_output.log >> errors.log ;
grep -m 1 "java.lang.ArrayIndexOutOfBoundsException" process_output.log >> errors.log ;
grep -m 1 "RuntimeError: java.lang.NoSuchMethodError" process_output.log >> errors.log ;
grep -m 1 "unzip: cannot find or open model.jar, model.jar.zip or model.jar.ZIP" process_output.log >> errors.log ;
grep -m 1 "ray.memory_monitor.RayOutOfMemoryError" process_output.log >> errors.log ;
grep -m 1 "FileNotFoundError: [Errno 2] No such file or directory: 'database/db.properties'" process_output.log >> errors.log ;
grep -m 1 "killed training" process_output.log >> errors.log ;
grep -m 1 "Job running for more than 24 hours, job is killed" process_output.log >> errors.log ;
grep -m 1 "Job crashed more than 3 times, job is killed" process_output.log >> errors.log