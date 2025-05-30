import random
from collections import defaultdict
from typing import Dict, List

from NoiseConfigs.utilsFunctions import UtilsFunc
from config import Config
from controllers.finalChoiceByAttenuationNoise import FinalChoiceByAttenuationNoise
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
from utils.enums import Layer
import sys
import os

sys.path.append(os.path.abspath("E:/VANET - Copy/NoiseConfigs"))


def yellow_bg(text):
    return f"\033[43m{text}\033[0m"

def red_bg(text):
    return f"\033[41m{text}\033[0m"

def blue_bg(text):
    return f"\033[44m{text}\033[0m"


# note: check again
def logAttenuation(attenuationList):
    sampleList = []
    for i in range(0, len(attenuationList)):
        sampleList.append(attenuationList[i])
    print(yellow_bg("attenuationList") + f":{len(sampleList)}")


def calcAttenuation(task, node, intersecting_partitions):
    """
    node : nearest_node or executor node
    """
    return UtilsFunc().path_loss_km_ghz(
        d_km=UtilsFunc().distance(task.creator.x, task.creator.y,
                                  node.x, node.y) / 1000,
        f_ghz=UtilsFunc().FREQUENCY_GH,
        n=UtilsFunc().get_max_urban_status(intersecting_partitions)
    ) + UtilsFunc().get_max_rain_attenuation(intersecting_partitions)


class Simulator:
    def __init__(self, loader: Loader, clock: Clock, cloud: CloudNode):
        self.metrics: MetricsController = MetricsController()
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

    # check traffic status
    def logTrafficStatus(self, partitions):
        for partition in partitions:
            print(
                f"Partition at ({partition.centerX}, {partition.centerY}): Traffic Status = {partition.trafficStatus}")

    def retransmission(self, zone_managers, current_time, partitions):
        tasks_to_retransmit = []
        for scheduled_time in list(self.retransmission_tasks.keys()):
            if scheduled_time <= current_time:
                tasks_to_retransmit.extend(self.retransmission_tasks.pop(scheduled_time))

        if tasks_to_retransmit:
            for task in tasks_to_retransmit:
                possible_zone_managers = self.find_zone_manager_offload_task(zone_managers, task, current_time)
                if self.choose_executor_and_assign(possible_zone_managers, task, partitions, current_time):
                    continue


    def find_zone_manager_offload_task(self, zone_managers, task, current_time):
        zone_manager_offload_task = []
        for zone_manager in zone_managers:
            # print(f"zone_manager:{zone_manager.zone}")
            if zone_manager.can_offload_task(task):
                # has_offloaded = True

                # assign task
                if hasattr(zone_manager, "propose_candidate"):
                    proposed_zone_manager, proposed_executor = zone_manager.propose_candidate(task,
                                                                                              current_time)
                    # print(f"proposed_executor:{proposed_executor}")
                else:
                    proposed_zone_manager = zone_manager
                    proposed_executor = zone_manager.offload_task(task, current_time)

                if proposed_executor not in zone_manager_offload_task:
                    if proposed_executor:
                        zone_manager_offload_task.append((proposed_zone_manager, proposed_executor))
        return zone_manager_offload_task

    def choose_executor_and_assign(self, zone_manager_offload_task, task, partitions, current_time):
        # if any ZM suggest any device to offload
        if len(zone_manager_offload_task) != 0:
            attenuationList = []

            for candidate in zone_manager_offload_task:
                zone_manager, candidate_executor = candidate
                # print(f"candidate_executor: {candidate_executor}")
                intersecting_partitions = UtilsFunc().find_line_intersections(
                    (task.creator.x, task.creator.y),
                    (candidate_executor.x, candidate_executor.y),
                    partitions
                )

                if len(intersecting_partitions) > 0 and not isinstance(candidate_executor, CloudNode):
                    attenuation = calcAttenuation(task, candidate_executor, intersecting_partitions)
                    # check : print(blue_bg(f"{attenuation}, {task.creator.x}, {task.creator.y}, {candidate_executor.x}, {candidate_executor.y}"))
                elif isinstance(candidate_executor, CloudNode):
                    # todo : find the nearest fog node and after that add a constraint latency for transmitting by fiber
                    # print(self.fixed_fog_nodes)
                    nearest_node_id = min(self.fixed_fog_nodes.keys(),
                                          key=lambda node_id: ((self.fixed_fog_nodes[node_id].x - task.creator.x) ** 2 +
                                                               (self.fixed_fog_nodes[node_id].y - task.creator.y) ** 2) ** 0.5)
                    nearest_node = self.fixed_fog_nodes[nearest_node_id]

                    attenuation = calcAttenuation(task, nearest_node, intersecting_partitions)

                else:
                    attenuation = 0
                    # locally offloading
                attenuationList.append((zone_manager, candidate_executor, attenuation))
            # logAttenuation(attenuationList)

            # todo: make decision to offload a task
            finalChoiceToOffload, plr = FinalChoiceByAttenuationNoise().makeFinalChoice(attenuationList,
                                                                                        task,
                                                                                        partitions,
                                                                                        Config.NoiseMethod.DEFAULT_METHOD)

            packetLossRandomNumber = random.randint(0, 100)

            # if finalChoiceToOffload:
            #     print(yellow_bg(f"finalChoiceToOffload:{finalChoiceToOffload}"))

            if finalChoiceToOffload:
                if packetLossRandomNumber < plr:
                    self.metrics.inc_packet_loss()
                    # print(blue_bg("----------------------------------------------------------------------------"))
                    # todo : add retransmission
                    chosen_zone_manager, chosen_executor, _ = finalChoiceToOffload
                    if task in chosen_executor.tasks:
                        chosen_executor.tasks.remove(task)

                    timeout_time = current_time + Config.SimulatorConfig.TIMEOUT_TIME
                    self.schedule_retransmission(task, timeout_time)

                else:
                    chosen_zone_manager, chosen_executor, _ = finalChoiceToOffload
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
                        chosen_zone_manager.agent.store_experience(state, action, reward, next_state, done=False)  # Store for training
                        chosen_zone_manager.agent.train()
            else:
                # note : there is not any device that meet noise problem, so they should retransmit too,
                #  but without any TIMEOUT, it will retry to exec in next step !

                # note todo: : it's good to make it run in the same time
                self.metrics.inc_no_device_found_to_run_becauseOf_Noise()

                timeout_time = current_time + 1
                self.schedule_retransmission(task, timeout_time)
        else:
            if Config.ZoneManagerConfig.DEFAULT_ALGORITHM == Config.ZoneManagerConfig.ALGORITHM_ONLY_FOG:
                self.metrics.inc_no_resource_found()
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

            # Update traffic status
            traffic_data = UtilsFunc.recognize_traffic_status(
                f"E:\pythonProject\VANET\SumoDividedByTime\Outputs2\dataInTime{int(self.clock.get_current_time())}.csv")
            partitions = UtilsFunc.load_partitions("generated_hex_partitions")
            for partition in partitions:
                partition.update_traffic_status(traffic_data)
            # self.logTrafficStatus(partitions)

            nodes_tasks = self.load_tasks(current_time)
            user_possible_zones = self.assign_mobile_nodes_to_zones(self.user_nodes, layer=Layer.USER)
            mobile_possible_zones = self.assign_mobile_nodes_to_zones(self.mobile_fog_nodes, layer=Layer.FOG)

            merged_possible_zones: Dict[str, List[ZoneManagerABC]] = {**user_possible_zones, **mobile_possible_zones}

            for creator_id, tasks in nodes_tasks.items():
                zone_managers = merged_possible_zones.get(creator_id, [])
                # print(f"zoneManagers : {zone_managers}")
                self.retransmission(zone_managers, current_time, partitions)

                for task in tasks:
                    self.metrics.inc_total_tasks()
                    # has_offloaded = False

                    zone_manager_offload_task = self.find_zone_manager_offload_task(zone_managers, task, current_time)
                    # print(blue_bg(f"{len(zone_manager_offload_task)}"))
                    self.choose_executor_and_assign(zone_manager_offload_task, task, partitions, current_time)

            self.update_graph()
            self.execute_tasks_for_one_step()
            self.metrics.flush()

            self.metrics.log_metrics()
        self.drop_not_completed_tasks()

    def load_tasks(self, current_time: float) -> Dict[str, List[Task]]:
        tasks: Dict[str, List[Task]] = defaultdict(list)
        for creator_id, creator_tasks in self.loader.load_nodes_tasks(current_time).items():
            creator = None
            if creator_id in self.user_nodes:
                creator = self.user_nodes[creator_id]
            elif creator_id in self.mobile_fog_nodes:
                creator = self.mobile_fog_nodes[creator_id]
            assert creator is not None

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
            tasks = node.execute_tasks(self.clock.get_current_time())
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
                if task.has_migrated:
                    self.metrics.inc_migration()
                if task.has_migrated and task.is_deadline_missed:
                    self.metrics.inc_migrate_and_miss()
                if task.is_deadline_missed:
                    print(blue_bg(f"{task.id}: {task.release_time}, {task.deadline}, {task.exec_time}, {task.finish_time}, {task.executor.id}"))
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

    def get_next_task(self):
        """Retrieve the next unprocessed task from the current simulation step."""
        current_time = self.clock.get_current_time()
        tasks = self.load_tasks(current_time)  # Get tasks at this step

        print(f"[DEBUG] Current Time: {current_time}, Total Tasks at this step: {sum(len(t) for t in tasks.values())}")

        for task_list in tasks.values():
            for task in task_list:
                print(f"[DEBUG] Checking Task {task.id} - Completed: {task.is_completed}, Executor: {task.executor}")
                if not task.is_completed and task.executor is None:
                    print(f"[DEBUG] Found Unprocessed Task: {task.id}")
                    return task  # Return the first unprocessed task

        print("[DEBUG] No Unprocessed Tasks Found")
        return None  # No available tasks

    def create_retransmitted_task(task: Task) -> Task:
        new_id = task.id + "_R"
        new_task = Task(
            id=new_id,
            deadline=task.deadline,
            exec_time=task.exec_time,
            power=task.power,
            creator=task.creator
        )
        return new_task
