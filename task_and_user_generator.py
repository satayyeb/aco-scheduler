import csv
import os
import random
import xml.etree.ElementTree as Et
from collections import defaultdict
from dataclasses import dataclass
from xml.dom import minidom

import matplotlib.pyplot as plt


class Config:
    CHUNK_SIZE = 1200  # Chunk size in seconds

    class TaskConfig:
        MIN_EXEC_TIME: float = 12.5  # Slightly increased execution times
        MAX_EXEC_TIME: float = 25.0  # Tasks take longer to complete
        MIN_POWER_CONSUMPTION: float = 1.0  # Higher power consumption than before
        MAX_POWER_CONSUMPTION: float = 3.5
        DEADLINE_MIN_FREE_TIME: float = 3.0  # Less deadline flexibility # note : next time make it a little bit more
        DEADLINE_MAX_FREE_TIME: float = 15.0

    class VehicleConfig:
        TASK_GENERATION_RATE: float = 0.35  # More frequent task generation
        FUCKED_UP_TASK_GENERATION_RATE: float = 0.55
        TRAFFIC_MIN_SPEED_THRESHOLD: float = 10  # Lowered speed, causing occasional congestion
        LANE_TRAFFIC_THRESHOLD: int = 15  # More vehicles per lane (moderate traffic)
        MAX_COMPUTATION_POWER: float = 7.0  # note : change it in feature change !
        MIN_COMPUTATION_POWER: float = 3.5
        COMPUTATION_POWER_ROUND_DIGIT: int = 2
        LOW_TRANSMISSION_POWER = 20  # todo : it's homogeneous right now and it's good to make it heterogeneous
        # MEDIUM_TRANSMISSION_POWER = 10
        HIGH_TRANSMISSION_POWER = 5
        # TRANSMISSION_LIST = [LOW_TRANSMISSION_POWER, MEDIUM_TRANSMISSION_POWER, HIGH_TRANSMISSION_POWER]

    class MobileFogConfig:
        MAX_COMPUTATION_POWER: float = 12.0  # Slightly reduced power in fog nodes
        MIN_COMPUTATION_POWER: float = 7.0
        COMPUTATION_POWER_ROUND_DIGIT: int = 2


@dataclass
class Vehicle:
    """Represents a mobile fog node in the network with a unique identifier and spatial coordinates."""

    id: str
    x: float
    y: float
    angle: float
    speed: float
    power: float
    type: str
    lane: str


@dataclass
class Task:
    id: str
    deadline: float
    exec_time: float  # The amount of time that this task required to execute.
    power: float  # The amount of power unit that this tasks consumes while executing.
    creator: str  # Thd id of the node who created the task.


class Generator:
    def __init__(self):
        self.current_chunk = 0
        self.current_vehicles = []
        self.current_tasks = []
        self.tasks_count_per_step = defaultdict(int)
        self.average_speed_per_step = defaultdict(float)
        self.total_task_power_per_step = defaultdict(float)

        # Create output directories
        os.makedirs("./data/vehicles", exist_ok=True)
        os.makedirs("./data/tasks", exist_ok=True)

    @staticmethod
    def get_chunk_number(step: int) -> int:
        return step // Config.CHUNK_SIZE

    def save_current_chunk(self, step: int):
        chunk_num = self.get_chunk_number(step)
        if chunk_num > self.current_chunk and (self.current_vehicles or self.current_tasks):
            self._save_vehicles_chunk()
            self._save_tasks_chunk()
            self.current_vehicles = []
            self.current_tasks = []
            self.current_chunk = chunk_num

    def _save_vehicles_chunk(self):
        root = Et.Element('fcd-export')
        root.set("version", "1.0")

        for time_data in self.current_vehicles:
            time_elem = Et.SubElement(root, 'timestep')
            time_elem.set('time', f"{time_data['step']}")
            for vehicle in time_data['vehicles']:
                v_elem = Et.SubElement(time_elem, 'vehicle')
                v_elem.set('id', vehicle.id)
                v_elem.set('x', f"{vehicle.x:.2f}")
                v_elem.set('y', f"{vehicle.y:.2f}")
                v_elem.set('angle', f"{vehicle.angle:.2f}")
                v_elem.set('speed', f"{vehicle.speed:.2f}")
                v_elem.set('lane', vehicle.lane)
                v_elem.set('type', vehicle.type)
                v_elem.set('power', f"{vehicle.power:.2f}")

        xml_str = minidom.parseString(Et.tostring(root)).toprettyxml(indent="    ")
        with open(f"./data/vehicles/chunk_{self.current_chunk}.xml", 'w', encoding='utf-8') as f:
            f.write(xml_str)

    def _save_tasks_chunk(self):
        root = Et.Element('fcd-export')
        root.set("version", "1.0")

        for time_data in self.current_tasks:
            time_elem = Et.SubElement(root, 'timestep')
            time_elem.set('time', f"{time_data['step']}")
            for task in time_data['tasks']:
                t_elem = Et.SubElement(time_elem, 'task')
                t_elem.set('id', task.id)
                t_elem.set('deadline', f"{task.deadline:.2f}")
                t_elem.set('exec_time', f"{task.exec_time:.2f}")
                t_elem.set('power', f"{task.power:.2f}")
                t_elem.set('creator', task.creator)

        xml_str = minidom.parseString(Et.tostring(root)).toprettyxml(indent="    ")
        with open(f"./data/tasks/chunk_{self.current_chunk}.xml", 'w', encoding='utf-8') as f:
            f.write(xml_str)

    def calculate_metrics(self, step: float, vehicles: list[Vehicle], tasks: list[Task]):
        """Calculate metrics for the current timestep."""
        # Count tasks for this step
        self.tasks_count_per_step[step] = len(tasks)

        # Calculate average speed of user nodes (PKW_special type)
        if vehicles:
            avg_speed = sum(v.speed for v in vehicles) / len(vehicles)
            self.average_speed_per_step[step] = round(avg_speed, 2)
        else:
            self.average_speed_per_step[step] = 0.0

        # Calculate total power of tasks for this step
        self.total_task_power_per_step[step] = round(sum(task.power for task in tasks), 2)

    @staticmethod
    def generate_one_step_task(step, vehicle, lane_counter):
        """Generate tasks for each mobile fog node."""
        exec_time = round(
            random.uniform(
                Config.TaskConfig.MIN_EXEC_TIME,
                Config.TaskConfig.MAX_EXEC_TIME,
            ),
            2
        )
        deadline_free = round(
            random.uniform(
                Config.TaskConfig.DEADLINE_MIN_FREE_TIME,
                Config.TaskConfig.DEADLINE_MAX_FREE_TIME,
            ),
            2
        )
        deadline = round(exec_time + deadline_free) + step
        power = round(
            random.uniform(
                Config.TaskConfig.MIN_POWER_CONSUMPTION,
                Config.TaskConfig.MAX_POWER_CONSUMPTION
            ),
            2
        )
        chance = random.random()
        threshold = Config.VehicleConfig.TASK_GENERATION_RATE
        if (
                lane_counter > Config.VehicleConfig.LANE_TRAFFIC_THRESHOLD or
                vehicle.speed < Config.VehicleConfig.TRAFFIC_MIN_SPEED_THRESHOLD
        ):
            threshold = Config.VehicleConfig.FUCKED_UP_TASK_GENERATION_RATE
        if chance > threshold:
            return None
        return Task(
            id=f"{vehicle.id}_{step}",
            deadline=deadline,
            exec_time=exec_time,
            power=power,
            creator=vehicle.id
        )

    def generate_one_step(self, step, time_data, seen_ids_power):
        """Generate vehicles for each mobile fog node."""
        current_vehicles = []
        current_tasks = []
        lane_counter = defaultdict(int)

        for vehicle in time_data.findall('vehicle'):
            v_id = vehicle.get('id')
            data = dict(
                id=v_id,
                x=float(vehicle.get('x')),
                y=float(vehicle.get('y')),
                angle=90 - float(vehicle.get('angle')),
                speed=float(vehicle.get('speed')),
                lane=vehicle.get('lane'),
                type=vehicle.get('type')
            )

            if v_id in seen_ids_power:
                power = seen_ids_power[v_id]
            elif vehicle.get('type') == "LKW_special":
                power = round(
                    random.uniform(
                        Config.MobileFogConfig.MIN_COMPUTATION_POWER,
                        Config.MobileFogConfig.MAX_COMPUTATION_POWER
                    ),
                    Config.MobileFogConfig.COMPUTATION_POWER_ROUND_DIGIT
                )
            elif vehicle.get('type') == "PKW_special":
                power = round(
                    random.uniform(
                        Config.VehicleConfig.MIN_COMPUTATION_POWER,
                        Config.VehicleConfig.MAX_COMPUTATION_POWER
                    ),
                    Config.MobileFogConfig.COMPUTATION_POWER_ROUND_DIGIT
                )
            else:
                continue

            seen_ids_power[v_id] = power
            data["power"] = power
            vehicle_obj = Vehicle(**data)
            current_vehicles.append(vehicle_obj)
            lane_counter[vehicle_obj.lane] += 1

            if task := self.generate_one_step_task(step, vehicle_obj, lane_counter[vehicle_obj.lane]):
                current_tasks.append(task)

        # Calculate metrics before saving the chunk
        self.calculate_metrics(step, current_vehicles, current_tasks)

        # Add current timestep data to the chunk
        self.current_vehicles.append({"step": step, "vehicles": current_vehicles})
        self.current_tasks.append({"step": step, "tasks": current_tasks})

        self.save_current_chunk(step)

        return seen_ids_power

    def generate_data(self, path: str):
        """Parse the time data from the given content."""
        with open(path, 'rb') as f:
            root = Et.parse(f).getroot()
        seen_ids_power = {}
        for time in root.findall('.//timestep'):
            step = round(float(time.get('time')))
            seen_ids_power = self.generate_one_step(step, time, seen_ids_power)

        # Save the last chunk if there's any data left
        if self.current_vehicles or self.current_tasks:
            self._save_vehicles_chunk()
            self._save_tasks_chunk()

    def save_metrics_to_csv(self, metrics_file: str):
        """Save the collected metrics to a CSV file."""
        all_steps = sorted(set(self.tasks_count_per_step.keys()) |
                           set(self.average_speed_per_step.keys()) |
                           set(self.total_task_power_per_step.keys()))

        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestep', 'task_count', 'average_speed', 'total_task_power'])
            for step in all_steps:
                writer.writerow([
                    f"{step:.2f}",
                    self.tasks_count_per_step[step],
                    self.average_speed_per_step[step],
                    self.total_task_power_per_step[step]
                ])

    def plot_metrics(self, output_file: str):
        """Create a visualization of the metrics."""
        steps = sorted(set(self.tasks_count_per_step.keys()) |
                       set(self.average_speed_per_step.keys()) |
                       set(self.total_task_power_per_step.keys()))
        task_counts = [self.tasks_count_per_step[step] for step in steps]
        avg_speeds = [self.average_speed_per_step[step] for step in steps]
        total_powers = [self.total_task_power_per_step[step] for step in steps]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        ax1.plot(steps, task_counts, 'b-', label='Tasks per Step')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Number of Tasks')
        ax1.set_title('Tasks Generated per Time Step')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(steps, avg_speeds, 'r-', label='Average Speed')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Speed')
        ax2.set_title('Average Speed of User Nodes per Time Step')
        ax2.grid(True)
        ax2.legend()

        ax3.plot(steps, total_powers, 'g-', label='Total Task Power')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Power Units')
        ax3.set_title('Total Power of Tasks per Time Step')
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()


def main(path: str):
    """Main function to run the task generator."""
    generator = Generator()
    generator.generate_data(path)
    generator.save_metrics_to_csv("./data/metrics.csv")
    generator.plot_metrics("./data/metrics_visualization.png")


if __name__ == '__main__':
    main("./simulation.out.xml")
