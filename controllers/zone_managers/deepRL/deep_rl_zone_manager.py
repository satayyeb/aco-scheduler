import numpy as np
from typing import Unpack

from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from controllers.zone_managers.deepRL.deep_rl_agent import DeepRLAgent
from controllers.zone_managers.deepRL.deep_rl_env import DeepRLEnvironment
from models.node.fog import FogLayerABC
from models.task import Task
from config import Config


class DeepRLZoneManager(ZoneManagerABC):
    """
    A zone manager that uses Deep Reinforcement Learning for task offloading.
    """

    def __init__(self, zone):
        super().__init__(zone)

        # Initialize Deep RL Environment and Agent
        # note: here removed
        # self.env = DeepRLEnvironment(simulator=None)  # Will be set in simulation
        self.agent = DeepRLAgent(state_dim=6, action_dim=3)  # 5 state features, 3 actions

        self.env = None

        # Load pre-trained model if available
        try:
            self.agent.load_model()
            print("[DeepRL] Loaded pre-trained model.")
        except:
            print("[DeepRL] No pre-trained model found, starting fresh.")

    def set_simulator(self, simulator):
        """
        Transfers the simulator reference to the environment and the agent.
        """
        self.env = DeepRLEnvironment(simulator)

    def can_offload_task(self, task: Task) -> bool:
        """
        Checks if there is an available node to offload the task.
        """
        available_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
        if self.env.simulator and self.env.simulator.cloud_node:
            available_nodes.append(self.env.simulator.cloud_node)
        return any(node.can_offload_task(task) for node in
                   available_nodes)
        # NOTE :I removed "or self.env.simulator.cloud_node.can_offload_task(task)"

    # note : not important function
    def assign_task(self, task: Task) -> FogLayerABC:
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        state = self.env._get_state(task)
        action = self.agent.select_action(state)

        if action == 0:
            candidate_executor = task.creator
        elif action == 1:
            candidate_executor = self._get_best_fog_node(task)
        else:
            candidate_executor = self.env.simulator.cloud_node

        return candidate_executor

    def propose_candidate(self, task: Task, current_time: float):
        """
        Uses Deep RL to decide where to offload a task.
        It just suggests a node and return (ZN, node)
        """
        state = self.env._get_state(task)
        action = self.agent.select_action(state)
        if action == 0:
            candidate_executor = task.creator
        elif action == 1:
            candidate_executor = self._get_best_fog_node(task)
        else:
            candidate_executor = self.env.simulator.cloud_node
        return self, candidate_executor

    def _get_best_fog_node(self, task):
        """
        Finds the best fog node to offload the task based on available resources.
        """
        # print("nooooo")
        fog_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
        fog_nodes = [node for node in fog_nodes if node.can_offload_task(task)]
        # note: that was min , i changed another one too
        return max(fog_nodes, key=lambda n: n.remaining_power, default=None)

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        """
        Updates the RL agent after each simulation step.
        """
        self.agent.train()  # Train the agent periodically
        if np.random.random() < 0.05:  # Update target network occasionally
            self.agent.update_target_network()
