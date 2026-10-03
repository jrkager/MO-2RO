from typing import Hashable

class Instance:

    scenarios = None

    # the four per-scenario weight vectors every problem in this repo defines:
    # (attribute, which objective it belongs to)
    WEIGHT_VECTORS = (("fs_weights_1", 1), ("ss_weights_1", 1),
                      ("fs_weights_2", 2), ("ss_weights_2", 2))

    def scale_objective_weights(self, obj1_factor: float = 1.0,
                                obj2_factor: float = 1.0):
        """Multiply every weight of objective 1 / 2 by a factor, in place.

        Both stages are scaled, so each objective is multiplied as a whole and the
        feasible set is untouched: the instance keeps its structure but the two
        objectives become deliberately asymmetric, which is what the slab
        orientation reacts to (see the runners' OBJ1_FACTORS / OBJ2_FACTORS grid).

        Apply this to a freshly built instance, BEFORE the reference point is
        computed and before the algorithm sees it -- it changes the scale of the
        objectives and hence the reference point, the hypervolume, the front sizes,
        and what a given slab width means.

        Only `self.scenarios` is touched; the class-level `nominal_scenario`
        template is shared between instances and must stay untouched. Products are
        kept exact (a non-integral factor yields non-integral weights).
        """
        if obj1_factor == 1.0 and obj2_factor == 1.0:
            return self
        for s in self.get_scenario_keys():
            sc = self.scenarios[s]
            for attr, obj in self.WEIGHT_VECTORS:
                factor = obj1_factor if obj == 1 else obj2_factor
                if factor != 1.0:
                    setattr(sc, attr, [w * factor for w in getattr(sc, attr)])
        return self

    def init_subset(self, mode: str = "nominal") -> set[Hashable]:
        """
        Initial scenario subset D for the master.
        :param mode:
            "nominal" -> {first scenario} (the historical default),
            "empty"   -> {} (the master then picks the worst scenario for some
                         feasible first-stage solution x0, see
                         HypervolumeMaster.get_worst_scenario),
            "cost"    -> structural worst-scenario guess from the cost data,
            "random"  -> one uniformly random scenario.
        """
        pass

    def get_scenario_keys(self) -> set[Hashable]:
        pass

    def get_first_objective(self, x, y, s):
        pass

    def get_second_objective(self, x, y, s):
        pass

    def add_first_stage_variables(self, model):
        pass

    def add_second_stage_variables(self, s, model, additional_index):
        pass

    def add_first_stage_constraints(self, x, model):
        pass

    def add_second_stage_constraints(self, x, y, s, model):
        """
        :param x: can be a first-stage solution object (fixed) or a gurobi variable dict
        :param y: gurobi variable dict for second-stage variables
        :param model:
        :return:
        """
        pass

    def get_first_stage_values(self, x):
        pass

    def get_best_ref_point(self) -> tuple:
        """
        Returns the ref point calculated with the best known method. For SEL and ASS it will be the enumerative approach.
        :return:
        """
        pass

    def info(self) -> str:
        pass
