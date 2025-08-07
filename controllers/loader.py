from collections import defaultdict
from typing import Type

from NoiseConfigs.utilsFunctions import UtilsFunc
from controllers.zone_managers.aco.zone_manager import AcoZoneManager
from controllers.zone_managers.base import ZoneManagerABC
from controllers.zone_managers.heuristic import HeuristicZoneManager
from controllers.zone_managers.random import RandomZoneManager
from controllers.zone_managers.hrl import HRLZoneManager
from controllers.zone_managers.only_cloud import OnlyCloudZoneManager
from controllers.zone_managers.only_fog import OnlyFogZoneManager
from controllers.zone_managers.deepRL.deep_rl_zone_manager import DeepRLZoneManager
from utils.xml_parser import *


class Loader:
    # TODO: Fill this map after adding zone managers completed.
    ALGORITHM_MAP: Dict[str, Type[ZoneManagerABC]] = {
        Config.ZoneManagerConfig.ALGORITHM_RANDOM: RandomZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_HEURISTIC: HeuristicZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_HRL: HRLZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_ONLY_CLOUD: OnlyCloudZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_ONLY_FOG: OnlyFogZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_DEEP_RL: DeepRLZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_ACO: AcoZoneManager,
    }

    def __init__(self, zone_file: str, fixed_fn_file: str, mobile_file: str, task_file: str, checkpoint_path: str):
        self.current_chunk = 0
        self.chunk_size = Config.CHUNK_SIZE
        self.zone_parser = ZoneSumoXMLParser(zone_file)
        self.fixed_fn_parser = FixedFogNodeSumoXMLParser(fixed_fn_file)
        self.mobile_chunk_path = mobile_file
        self.mobile_node_parser = MobileNodeSumoXMLParser(mobile_file, 0)
        self.task_chunk_path = task_file
        self.task_parser = TaskSumoXMLParser(task_file, 0)
        self.checkpoint_path = checkpoint_path

    def __load_next_chunk(self, time_step: float):
        if self.get_chunk(time_step - 1) != self.current_chunk and time_step != 0:
            self.mobile_node_parser = MobileNodeSumoXMLParser(self.mobile_chunk_path, self.get_chunk(time_step))
            self.task_parser = TaskSumoXMLParser(self.task_chunk_path, self.get_chunk(time_step))
            self.current_chunk = self.get_chunk(time_step)

    def get_chunk(self, time_step: float) -> int:
        return round(time_step) // self.chunk_size

    def load_zones(self) -> Dict[str, ZoneManagerABC]:
        zone_managers: Dict[str, ZoneManagerABC] = {}

        zones = self.zone_parser.parse()

        # note: removed!
        # partitions = UtilsFunc.load_partitions("generated_hex_partitions")
        # factory_partitions = self.findAllFactoryPartitions(partitions)

        for zone in zones:
            zone_manager_cls = self.ALGORITHM_MAP[Config.ZoneManagerConfig.DEFAULT_ALGORITHM]

            zone_manager_obj = zone_manager_cls(zone)

            # If using DeepRL, set the simulator reference
            # if isinstance(zone_manager_obj, DeepRLZoneManager):
            #     zone_manager_obj.env.simulator = self  # ✅ Attach the simulator to the DeepRL environment

            zone_managers[zone.id] = zone_manager_obj

            if isinstance(zone_manager_obj, HRLZoneManager):
                zone_manager_obj.load_checkpoint(self.checkpoint_path)

        return zone_managers

    def load_fixed_zones(self) -> Dict[str, FixedFogNode]:
        fixed_fog_nodes: Dict[str, FixedFogNode] = {}

        for fixed_node in self.fixed_fn_parser.parse():
            fixed_fog_nodes[fixed_node.id] = fixed_node
        return fixed_fog_nodes

    def load_mobile_fog_nodes(self, time_step: float) -> Dict[str, MobileFogNode]:
        self.__load_next_chunk(time_step)
        mobile_fog_nodes: Dict[str, MobileFogNode] = {}

        if time_step < Config.SimulatorConfig.SIMULATION_DURATION - 1:
            for mobile_node in self.mobile_node_parser.parse()[time_step][1]:
                mobile_fog_nodes[mobile_node.id] = mobile_node

        return mobile_fog_nodes

    def load_user_nodes(self, time_step: float) -> Dict[str, UserNode]:
        self.__load_next_chunk(time_step)
        user_fog_nodes: Dict[str, UserNode] = {}

        if time_step < Config.SimulatorConfig.SIMULATION_DURATION:
            for user_node in self.mobile_node_parser.parse()[time_step][0]:
                user_fog_nodes[user_node.id] = user_node
        return user_fog_nodes

    def load_nodes_tasks(self, time_step: float) -> Dict[str, List[Task]]:
        self.__load_next_chunk(time_step)
        tasks: Dict[str, List[Task]] = defaultdict(list)

        for task in self.task_parser.parse().get(time_step, []):
            tasks[task.creator_id].append(task)
        return tasks

    # def findAllFactoryPartitions(self, partitions):
    #     temp = []
    #     for partition in partitions:
    #         if partition.is_factory:
    #             temp.append(partition)
    #     return temp
