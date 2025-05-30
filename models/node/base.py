from __future__ import annotations

import abc
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from config import Config
from models.base import ModelBaseABC
from utils.enums import Layer
from utils.distance import get_distance

def blue_bg(text):
    return f"\033[44m{text}\033[0m"

def findExecTimeInEachKindOfNode(task, executor=None):
    from models.node.user import UserNode
    from models.node.cloud import CloudNode
    from models.node.fog import FixedFogNode
    from models.node.fog import MobileFogNode

    taskExecutor = task.executor
    if executor:
        taskExecutor = executor
    if isinstance(taskExecutor, UserNode):
        # print("UserNode()")
        return task.real_exec_time(executor=taskExecutor)
    elif isinstance(taskExecutor, CloudNode):
        # print("CloudNode()")
        return task.real_exec_time(executor=taskExecutor) / 2
    elif isinstance(taskExecutor, FixedFogNode):
        # print("FixedFogNode()")
        return task.real_exec_time(executor=taskExecutor) / 2
    elif isinstance(taskExecutor, MobileFogNode):
        # print("MobileFogNode()")
        return task.real_exec_time(executor=taskExecutor) / 1.5
    else:
        print("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrorr")
        return -1

    # return real_exec_time_base


@dataclass
class NodeABC(ModelBaseABC, abc.ABC):
    """
    Represents any computational resource in the system. Nodes can belong to different layers,
    such as User, Fog, or Cloud.
    """

    x: float = 0
    y: float = 0
    power: float = 0  # The computational power available at this node.

    radius: float = 0  # The radius that this node can cover.

    remaining_power: float = 0  # The amount of computational resourced left after executing current tasks.
    tasks: Deque = field(default_factory=deque)  # The list of tasks that are currently offloaded in this node.
    finished_tasks: Deque = field(default_factory=deque)  # The list of tasks that are finished executing in this node.

    def can_offload_task(self, task) -> bool:
        """Checks whether the task can be offloaded in this node."""
        # print("----------------------------------test----------------------------------")
        if len(self.tasks) >= self.max_tasks_queue_len:
            # print(blue_bg(f"max_tasks_queue_len"))
            return False
        # if 1.0 - self.remaining_power / self.power > self.used_power_limit:
        #     # print(blue_bg(f"test????????????"))
        #     return False
        task_power = task.power
        if self.id == task.creator.id and self.layer == Layer.USER:
            task_power *= Config.UserNodeConfig.LOCAL_OFFLOAD_POWER_OVERHEAD
        if task_power > self.remaining_power:
            # print(blue_bg(f"remaining_power"))
            return False
        if get_distance(self.x, self.y, task.creator.x, task.creator.y) > self.radius:
            # print(blue_bg(f"distance"))
            return False
        return True

    def assign_task(self, task, current_time: float) -> None:
        """Offload a task in the current node."""
        # print(f"task:{task}")
        # note: i think below code has been checked
        # if not self.can_offload_task(task):
        #     raise Exception

        self.tasks.append(task)
        task.executor = self

        task_power = task.power
        if self.id == task.creator.id and self.layer == Layer.USER:
            task_power *= Config.UserNodeConfig.LOCAL_OFFLOAD_POWER_OVERHEAD
        self.remaining_power -= task_power
        task.start_time = current_time
        # task.start_time = task.release_time

    def execute_tasks(self, current_time: float) -> list:
        """Execute current tasks and return all completed tasks."""
        remaining_tasks = deque()
        finished_tasks = deque()

        while self.tasks:
            task = self.tasks.popleft()
            # print(f"task: {task}")
            # print(f"[DEBUG] Processing task {task.id}: start_time={task.start_time}, current_time={current_time}, base_exec_time={task.real_exec_time}")

            task.real_exec_time_base = findExecTimeInEachKindOfNode(task)
            # if task.id == "PKW10_2":
            #     print(f"[DEBUG]{task.id}: real_exec_time_base={task.real_exec_time_base}")

            if current_time - task.release_time >= task.real_exec_time_base:
                # if task.id == "PKW10_2":
                #     print(f"[DEBUG]{task.id} +++: real_exec_time_base={task.real_exec_time_base}")
                finished_tasks.append(task)

                task_power = task.power
                if self.id == task.creator.id and self.layer == Layer.USER:
                    task_power *= Config.UserNodeConfig.LOCAL_OFFLOAD_POWER_OVERHEAD
                self.remaining_power += task_power
            else:
                remaining_tasks.append(task)

        self.tasks = remaining_tasks
        self.finished_tasks = self.finished_tasks + finished_tasks

        finished_tasks = []
        remaining_tasks = deque()
        while self.finished_tasks:

            task = self.finished_tasks.popleft()
            # if task.id == "PKW10_2":
            #     print(f"[DEBUG]{task.id} &&&&&&&: real_exec_time_base={task.real_exec_time_base}")
            distance = task.get_creator_and_executor_distance()
            if not task:
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            real_exec_time = task.real_exec_time_base + \
                             distance * 2 * Config.TaskConfig.PACKET_COST_PER_METER + \
                             distance * Config.TaskConfig.TASK_COST_PER_METER
            print(blue_bg(f"transmission delay: {distance * 2 * Config.TaskConfig.PACKET_COST_PER_METER + distance * Config.TaskConfig.TASK_COST_PER_METER}, distance: {distance}, task.real_exec_time_base: {task.real_exec_time_base}, executor: {task.executor}"))
            if self.layer == Layer.CLOUD:
                real_exec_time += Config.TaskConfig.CLOUD_PROCESSING_OVERHEAD
            elif task.has_migrated:
                real_exec_time += Config.TaskConfig.MIGRATION_OVERHEAD * distance

            # print(f"[DEBUG] Re-checking task {task.id}: new_exec_time={real_exec_time}, elapsed={current_time - task.start_time}")

            if current_time - task.start_time >= real_exec_time:
                task.finish_time = task.start_time + real_exec_time
                finished_tasks.append(task)
            else:
                remaining_tasks.append(task)

        self.finished_tasks = remaining_tasks

        # print(f"finished_tasks: {finished_tasks}")
        return finished_tasks

    @property
    @abc.abstractmethod
    def layer(self) -> Layer:
        """The layer to which this node belongs (User, Fog, or Cloud)."""
        raise NotImplemented

    @property
    @abc.abstractmethod
    def max_tasks_queue_len(self) -> int:
        """The max number of tasks that this node can process in parallel."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def used_power_limit(self) -> float:
        """The maximum power that the node can use to execute its tasks."""
        raise NotImplementedError


@dataclass
class MobileNodeABC(NodeABC, abc.ABC):
    """
    Represents any computational resource in the system which can move from one location to another. Mobile nodes may
    belong to different layers, such as User or Fog.
    """

    speed: float = 0
    angle: float = 0
