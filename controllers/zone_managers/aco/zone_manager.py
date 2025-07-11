from typing import Unpack

from controllers.zone_managers.aco.aco import ACO
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate

from models.node.base import NodeABC, findExecTimeInEachKindOfNode, calculate_distance
from models.task import Task


class AcoZoneManager(ZoneManagerABC):

    def can_offload_task(self, task: Task) -> bool:

        self.all_possible_nodes: list[NodeABC] = []
        for node in {**self.fixed_fog_nodes, **self.mobile_fog_nodes}.values():
            if node.can_offload_task(task):
                self.all_possible_nodes.append(node)

        if task.creator.can_offload_task(task):
            self.all_possible_nodes.insert(0, task.creator)

        if len(self.all_possible_nodes) == 0:
            return False

        from main import cloud
        if cloud.can_offload_task(task):
            self.all_possible_nodes.append(cloud)

        return True

    def assign_task(self, task: Task) -> NodeABC:
        distances = []
        exec_times = []
        for node in self.all_possible_nodes:
            distances.append(calculate_distance(task.creator.x, task.creator.y, node.x, node.y))
            exec_times.append(findExecTimeInEachKindOfNode(task, node))
        index = ACO(distances, exec_times).run()
        return self.all_possible_nodes[index]

    def offload_task(self, task: Task, current_time: float) -> NodeABC:
        return super().offload_task(task, current_time)

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        pass
