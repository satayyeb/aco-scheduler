import random
from collections import defaultdict
from typing import Dict, List

from matplotlib import pyplot as plt

from config import Config
from controllers.finalChoice import FinalChoice
from controllers.loader import Loader
from controllers.metric import MetricsController
from controllers.zone_managers.base import ZoneManagerABC
from controllers.zone_managers.deepRL.deep_rl_zone_manager import DeepRLZoneManager
from models.node.base import MobileNodeABC, NodeABC
from models.node.cloud import CloudNode
from models.node.fog import FixedFogNode
from models.node.fog import MobileFogNode
from models.node.user import UserNode
from models.task import Task
from utils.clock import Clock
from utils.enums import Layer, VehicleApplicationType


def yellow_bg(text):
    return f"\033[43m{text}\033[0m"


def red_bg(text):
    return f"\033[41m{text}\033[0m"


def blue_bg(text):
    return f"\033[44m{text}\033[0m"


class Simulator:
    def __init__(self, loader: Loader, clock: Clock, cloud: CloudNode):
        self.metrics: MetricsController = MetricsController(clock)
        self.loader: Loader = loader
        self.cloud_node: CloudNode = cloud
        self.zone_managers: Dict[str, ZoneManagerABC] = {}
        self.fixed_fog_nodes: Dict[str, FixedFogNode] = {}
        self.mobile_fog_nodes: Dict[str, MobileFogNode] = {}
        self.user_nodes: Dict[str, UserNode] = {}
        self.clock: Clock = clock
        self.task_zone_managers: Dict[str, ZoneManagerABC] = {}
        self.retransmission_tasks: Dict[float, List[Task]] = {}

    def init_simulation(self):
        self.clock.set_current_time(0)
        self.zone_managers = self.loader.load_zones()
        self.fixed_fog_nodes = self.loader.load_fixed_zones()
        self.assign_fixed_nodes()
        self.update_mobile_fog_nodes_coordinate()
        self.update_user_nodes_coordinate()
        # For zone managers that use deep RL, the simulator reference is set.
        for zm in self.zone_managers.values():
            if hasattr(zm, "set_simulator"):
                zm.set_simulator(self)

    def schedule_retransmission(self, task: Task, scheduled_time: float):
        if scheduled_time not in self.retransmission_tasks:
            self.retransmission_tasks[scheduled_time] = []
        self.retransmission_tasks[scheduled_time].append(task)

    def assign_fixed_nodes(self):
        for z_id, zone_manager in self.zone_managers.items():
            fixed_nodes: List[FixedFogNode] = []
            for n_id, fixed_node in self.fixed_fog_nodes.items():
                if zone_manager.zone.is_in_coverage(fixed_node.x, fixed_node.y):
                    fixed_nodes.append(fixed_node)
            zone_manager.add_fixed_fog_nodes(fixed_nodes)

    def retransmission(self, zone_managers, current_time):
        tasks_to_retransmit = []
        for scheduled_time in list(self.retransmission_tasks.keys()):
            if scheduled_time <= current_time:
                tasks_to_retransmit.extend(self.retransmission_tasks.pop(scheduled_time))

        if tasks_to_retransmit:
            for task in tasks_to_retransmit:
                possible_zone_managers = self.find_zone_manager_offload_task(zone_managers, task, current_time)
                if self.choose_executor_and_assign(possible_zone_managers, task, current_time):
                    continue

    def find_zone_manager_offload_task(self, zone_managers, task, current_time):
        zone_manager_offload_task = []
        for zone_manager in zone_managers:
            # print(f"zone_manager:{zone_manager.zone}")
            if zone_manager.can_offload_task(task):

                # assign task
                if hasattr(zone_manager, "propose_candidate"):
                    proposed_zone_manager, proposed_executor = zone_manager.propose_candidate(task, current_time)
                else:
                    proposed_zone_manager = zone_manager
                    proposed_executor = zone_manager.offload_task(task, current_time)

                if proposed_executor not in zone_manager_offload_task:
                    if proposed_executor:
                        zone_manager_offload_task.append((proposed_zone_manager, proposed_executor))
        return zone_manager_offload_task

    def choose_executor_and_assign(self, zone_manager_offload_task, task, current_time) -> None:
        if task.priority == VehicleApplicationType.CRUCIAL:
            while not task.creator.can_offload_task(task):
                preempted_task = task.creator.preempt_low_task()
                self.schedule_retransmission(preempted_task, current_time + 1)
                self.metrics.inc_preemption()
            self.metrics.inc_node_tasks(task.creator.id)
            task.creator.assign_task(task, current_time)
            return

        # if any ZM suggest any device to offload
        if len(zone_manager_offload_task) != 0:
            finalCandidates = []

            for candidate in zone_manager_offload_task:
                finalCandidates.append(candidate)

            finalChoiceToOffload = FinalChoice().makeFinalChoice(
                finalCandidates,
                Config.FinalDeciderMethod.DEFAULT_METHOD,
            )

            # if finalChoiceToOffload:
            #     print(yellow_bg(f"finalChoiceToOffload:{finalChoiceToOffload}"))

            chosen_zone_manager, chosen_executor = finalChoiceToOffload
            self.task_zone_managers[task.id] = chosen_zone_manager
            self.metrics.inc_node_tasks(chosen_executor.id)
            if isinstance(chosen_zone_manager, DeepRLZoneManager):
                state = chosen_zone_manager.env._get_state(task)  # Get current system state
                # print(blue_bg(f"------------chosen_executor: {chosen_executor}------------\n------------task: {task}------------"))
                reward, action = chosen_zone_manager.env._compute_reward2(task, chosen_executor)
                if not chosen_executor.can_offload_task(task) and (reward > -100):
                    reward = -100
                    timeout_time = current_time + 1
                    self.schedule_retransmission(task, timeout_time)
                elif reward < -100:
                    timeout_time = current_time + 1
                    self.schedule_retransmission(task, timeout_time)
                else:
                    chosen_executor.assign_task(task, current_time)

                # if reward < 0:
                #     print(red_bg(f"reward: --- {reward} --- {task.id}, {chosen_executor.id}"))
            else:
                chosen_executor.assign_task(task, current_time)

            # if reward < 0:
            #     print(chosen_executor.remaining_power)
            if isinstance(chosen_zone_manager, DeepRLZoneManager):
                next_state = chosen_zone_manager.env._get_state(task)
                chosen_zone_manager.agent.store_experience(state, action, reward, next_state,
                                                           done=False)  # Store for training
                chosen_zone_manager.agent.train()

        else:
            if task.creator.can_offload_task(task):
                task.creator.assign_task(task, current_time)
                self.metrics.inc_local_execution()
            else:
                self.offload_to_cloud(task, current_time)

    def start_simulation(self):
        self.init_simulation()
        while (current_time := self.clock.get_current_time()) < Config.SimulatorConfig.SIMULATION_DURATION:
            print(red_bg(f"current_time:{current_time}"))

            nodes_tasks = self.load_tasks(current_time)
            user_possible_zones = self.assign_mobile_nodes_to_zones(self.user_nodes, layer=Layer.USER)
            mobile_possible_zones = self.assign_mobile_nodes_to_zones(self.mobile_fog_nodes, layer=Layer.FOG)

            merged_possible_zones: Dict[str, List[ZoneManagerABC]] = {**user_possible_zones, **mobile_possible_zones}

            for creator_id, tasks in nodes_tasks.items():
                zone_managers = merged_possible_zones.get(creator_id, [])
                # print(f"zoneManagers : {zone_managers}")
                self.retransmission(zone_managers, current_time)

                for task in tasks:
                    self.metrics.inc_total_tasks()

                    zone_manager_offload_task = self.find_zone_manager_offload_task(zone_managers, task, current_time)
                    # check: this function
                    self.choose_executor_and_assign(zone_manager_offload_task, task, current_time)

            self.update_graph()
            self.execute_tasks_for_one_step()
            self.metrics.flush()

            self.metrics.log_metrics()
        self.drop_not_completed_tasks()

        self.plot_deadline_misses()
        self.plot_preemption()

    def plot_deadline_misses(self):
        print('plotting deadline misses')
        x_values, y_values = zip(*self.metrics.deadline_miss_history)
        plt.plot(x_values, y_values)
        plt.xlabel('Timestep')
        plt.ylabel('# Deadline misses')
        plt.title('Deadline miss history')
        plt.grid(True)
        plt.savefig('deadline.png')
        plt.show()

    def plot_preemption(self):
        print('plotting preemption')
        x_values, y_values = zip(*self.metrics.preemption_history)
        plt.plot(x_values, y_values)
        plt.xlabel('Timestep')
        plt.ylabel('# Preemption')
        plt.title('Preemption history')
        plt.grid(True)
        plt.savefig('preemption.png')
        plt.show()

    def load_tasks(self, current_time: float) -> Dict[str, List[Task]]:
        tasks: Dict[str, List[Task]] = defaultdict(list)
        for creator_id, creator_tasks in self.loader.load_nodes_tasks(current_time).items():
            creator = None
            if creator_id in self.user_nodes:
                creator = self.user_nodes[creator_id]
            elif creator_id in self.mobile_fog_nodes:
                creator = self.mobile_fog_nodes[creator_id]

            if creator is None:
                print(f"there is no {creator}\n")
            else:
                for task in creator_tasks:
                    task.creator = creator
                    tasks[creator_id].append(task)
        return tasks

    def execute_tasks_for_one_step(self):
        executed_tasks: List[Task] = []
        merged_nodes: Dict[str, NodeABC] = {
            **self.mobile_fog_nodes,
            **self.user_nodes,
            self.cloud_node.id: self.cloud_node,
        }
        for node_id, node in merged_nodes.items():
            tasks = node.execute_tasks(self.clock.get_current_time(), self.fixed_fog_nodes)
            # if tasks:
            #     print(yellow_bg(f"tasks :{tasks}"))
            executed_tasks.extend(tasks)
            for task in tasks:
                zone_manager = self.task_zone_managers.get(task.id)
                if zone_manager:
                    zone_manager.update(current_task=task)
                    all_fog_nodes = {**zone_manager.fixed_fog_nodes, **zone_manager.mobile_fog_nodes}
                    loads = [len(node.tasks) for node in all_fog_nodes.values() if node.can_offload_task(task)]
                    if loads:
                        min_load = min(loads)
                        max_load = max(loads)
                        self.metrics.inc_task_load_diff(task.id, min_load, max_load)

                if task.creator.id == task.executor.id:
                    self.metrics.inc_local_execution()
                elif isinstance(task.executor, FixedFogNode) or isinstance(task.executor, MobileFogNode):
                    self.metrics.inc_fog_execution()
                elif isinstance(task.executor, CloudNode):
                    self.metrics.inc_cloud_tasks()
                # if task.has_migrated:
                #     self.metrics.inc_migration()
                # if task.has_migrated and task.is_deadline_missed:
                #     self.metrics.inc_migrate_and_miss()
                if task.is_deadline_missed:
                    # print(blue_bg(
                    #     f"{task.id}: release_time:{task.release_time}, deadline:{task.deadline}, exec_time:{task.exec_time}, finish_time:{task.finish_time}, {task.executor.id}, {task.dataSize}, diff:{task.finish_time-task.deadline}"))
                    self.metrics.inc_deadline_miss()
                else:
                    self.metrics.inc_completed_task()

    def update_graph(self):
        self.clock.tick()
        self.update_user_nodes_coordinate()
        self.update_mobile_fog_nodes_coordinate()

    def offload_to_cloud(self, task: Task, current_time: float):
        if self.cloud_node.can_offload_task(task):
            self.cloud_node.assign_task(task, current_time)
            self.metrics.inc_cloud_tasks()
        else:
            self.metrics.inc_no_resource_found()

    def assign_mobile_nodes_to_zones(
            self,
            mobile_nodes: dict[str, MobileNodeABC],
            layer: Layer
    ) -> Dict[str, List[ZoneManagerABC]]:

        nodes_possible_zones: Dict[str, List[ZoneManagerABC]] = defaultdict(list)
        for z_id, zone_manager in self.zone_managers.items():
            nodes: List[MobileNodeABC] = []
            for n_id, mobile_node in mobile_nodes.items():
                if zone_manager.zone.is_in_coverage(mobile_node.x, mobile_node.y):
                    nodes.append(mobile_node)
                    nodes_possible_zones[n_id].append(zone_manager)
            if layer == Layer.FOG:
                zone_manager.set_mobile_fog_nodes(nodes)
        return nodes_possible_zones

    def update_mobile_fog_nodes_coordinate(self) -> None:
        new_nodes_data = self.loader.load_mobile_fog_nodes(self.clock.get_current_time())
        self.mobile_fog_nodes = self.update_nodes_coordinate(self.mobile_fog_nodes, new_nodes_data)

    def update_user_nodes_coordinate(self) -> None:
        new_nodes_data = self.loader.load_user_nodes(self.clock.get_current_time())
        self.user_nodes = self.update_nodes_coordinate(self.user_nodes, new_nodes_data)

    @staticmethod
    def update_nodes_coordinate(old_nodes: dict[str, MobileNodeABC], new_nodes: dict[str, MobileNodeABC]):
        data: Dict[str, MobileNodeABC] = {}
        for n_id, new_node in new_nodes.items():
            if n_id not in old_nodes:
                node = new_node
            else:
                node = old_nodes[n_id]
                node.x = new_node.x
                node.y = new_node.y
                node.angle = new_node.angle
                node.speed = new_node.speed
            data[n_id] = node
        return data

    def drop_not_completed_tasks(self) -> List[Task]:
        left_tasks: list[Task] = []
        merged_nodes: Dict[str, NodeABC] = {
            **self.mobile_fog_nodes,
            **self.user_nodes,
            self.cloud_node.id: self.cloud_node,
        }

        for node_id, node in merged_nodes.items():
            left_tasks.extend(node.tasks)
            for i in range(len(node.tasks)):
                self.metrics.inc_deadline_miss()
        return left_tasks
