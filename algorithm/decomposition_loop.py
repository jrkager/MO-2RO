from .interfaces import gp, GRB, time, os, List, Dict, plt
from .interfaces import MasterInterface, TerminationException, Instance, Type, get_print_function

def run(problem_instance:Instance, master:MasterInterface,
        target_gap:float,
        master_gap:float=0.0,
        logfile=None):
    """

    :param master:
    :param subproblem_mesasure:
    :param target_gap:
    :param logfile:
    :return:
    """
    myprint = get_print_function(logfile)
    myprint(f"Running with master {master.__class__.__name__}, target gap {target_gap}")
    myprint(f" on instance: {problem_instance.__class__.__name__}.")
    myprint(f" Instance details: {problem_instance.info()}")
    master.myprint = myprint
    master.print_config()

    x_tilde = None
    timings = {"master": 0, "eval": 0}

    terminate = False
    iteration = 1
    inner_iteration = 1
    iteration_offset = len(master.D)

    while not terminate:
        # master
        if len(master.D) - iteration_offset + 1 != iteration:
            iteration += 1
            iteration_offset = len(master.D) - iteration + 1
            inner_iteration = 1
        myprint(f"\n=== Iteration {len(master.D) - iteration_offset + 1}.{inner_iteration} ===")
        myprint("Solving master with D =", master.D)
        starttime = time.process_time()
        x_tilde, D_critical, z_tilde = master.solve_master(master_gap=master_gap)
        timings["master"] += time.process_time() - starttime

        # D_critical is the set of scenarios whose z-constraint is tight in the master
        # (within Gurobi's FeasibilityTol). It is informational only: the algorithm's
        # decisions are taken in eval_and_react from the AV/HV values of ALL
        # scenarios. (An earlier version re-evaluated every critical scenario here
        # with the epsilon-constraint method just to trim this printout -- one full
        # front computation per critical scenario per iteration, counted in neither
        # timer -- and compared the result with z_tilde without a tolerance.)
        myprint(f" Master solution = {x_tilde}. z_tilde = {z_tilde:.2f}, D_critical (tight in master) = {D_critical}")


        # # subproblem
        # starttime = time.process_time()
        # worst_scenarios, val = get_worst_scenarios(problem_instance, subproblem_mesasure, x_tilde, master)
        # timings["subproblem"] += time.process_time() - starttime
        # evaluate
        try:
            starttime = time.process_time()
            master.eval_and_react(target_gap, timings=timings)
            timings["eval"] += time.process_time() - starttime
        except TerminationException as ex:
            myprint(ex)
            terminate = True

        inner_iteration += 1

    # Human-readable timing summary so every log (not just the experiment runner)
    # ends with the runtime spent in each phase and the total.
    total_solve = sum(timings.values())
    myprint("\n" + "-" * 40)
    myprint("Solve-time breakdown (CPU / process_time):")
    for k, v in timings.items():
        myprint(f"  {k:<8s}: {v:8.3f}s")
    myprint(f"  {'total':<8s}: {total_solve:8.3f}s")

    return x_tilde, timings