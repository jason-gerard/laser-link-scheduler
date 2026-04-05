from .base_scheduler import BaseScheduler
from .alternating import AlternatingScheduler
from .laser_link import LaserLinkScheduler
from .fair_contact_plan import FairContactPlan
from .random_scheduler import RandomScheduler
from .path_solver import PathSchedulerModel
from .milp_lls import LLSModel
from .energy_aware import EnergyAware
from .lifespan_aware import LifespanAware
from .brute_force import BruteForceScheduler

__all__ = [
    "BaseScheduler",
    "LaserLinkScheduler",
    "RandomScheduler",
    "AlternatingScheduler",
    "BruteForceScheduler",
    "FairContactPlan",
    "EnergyAware",
    "LifespanAware",
    "PathSchedulerModel",
    "LLSModel",
]

SCHEDULER_REGISTER: dict[str, BaseScheduler] = {
    "lls": LaserLinkScheduler(),
    "lls_pat_unaware": LaserLinkScheduler(should_bypass_retargeting_time=True),
    "lls_mip": LLSModel(is_mip=True),
    "lls_lp": LLSModel(is_mip=False),
    "lls_path": PathSchedulerModel(),
    "fcp": FairContactPlan(),
    "random": RandomScheduler(),
    "alternating": AlternatingScheduler(),
    "energy_aware": EnergyAware(),
    "energy_aware_pat_unaware": EnergyAware(
        should_bypass_retargeting_time=True
    ),
    "lifespan_aware": LifespanAware(),
}
