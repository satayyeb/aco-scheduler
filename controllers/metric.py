from collections import defaultdict


class MetricsController:
    """Gathers and store all statistics metrics in our system."""

    def __init__(self):
        # General Metrics
        self.migrations_count = 0  # Total number of migrations happened in system.
        self.deadline_misses = 0  # Total number of deadline misses happened in system.
        self.no_resource_found = 0  # Total number of tasks that did not find resource to execute in system.
        self.migrate_and_miss = 0
        self.local_execution = 0
        self.total_tasks = 0  # Total number of tasks processed in system.
        self.cloud_tasks = 0  # Total number of tasks offloaded to cloud server.
        self.completed_tasks = 0  # Total number of tasks completed in system.

        # Per Node Metrics
        self.node_task_counts: dict[str, int] = defaultdict(int)

        # Per Step Metrics
        self.migration_counts_per_step: list[int] = []
        self.deadline_misses_per_step: list[int] = []
        self.completed_task_per_step: list[int] = []
        self.current_step_migrations = 0
        self.current_step_deadline_misses = 0
        self.current_step_completed_tasks = 0

        # Per task metrics
        self.task_load_diff: dict[int, tuple[float, float]] = {}

    def inc_task_load_diff(self, task_id:int, min_load: float, max_load: float):
        self.task_load_diff[task_id] = (min_load, max_load)

    def inc_local_execution(self):
        self.local_execution += 1

    def inc_migrate_and_miss(self):
        self.migrate_and_miss += 1

    def inc_completed_task(self):
        self.current_step_completed_tasks += 1
        self.completed_tasks += 1

    def inc_migration(self):
        self.current_step_migrations += 1
        self.migrations_count += 1

    def inc_no_resource_found(self):
        self.current_step_deadline_misses += 1
        self.deadline_misses += 1
        self.no_resource_found += 1

    def inc_deadline_miss(self):
        self.current_step_deadline_misses += 1
        self.deadline_misses += 1

    def inc_total_tasks(self):
        self.total_tasks += 1

    def inc_node_tasks(self, node_id: str):
        self.node_task_counts[node_id] += 1

    def inc_cloud_tasks(self):
        self.cloud_tasks += 1

    def flush(self):
        self.migration_counts_per_step.append(self.current_step_migrations)
        self.deadline_misses_per_step.append(self.current_step_deadline_misses)
        self.completed_task_per_step.append(self.current_step_completed_tasks)
        self.current_step_deadline_misses = 0
        self.current_step_migrations = 0
        self.current_step_completed_tasks = 0

    def log_metrics(self):
        print("Metrics:")
        print(f"\tTotal migrations: {self.migrations_count}")
        print(f"\tTotal deadline misses: {self.deadline_misses}")
        print(f"\tTotal migrate and misses: {self.migrate_and_miss}")
        print(f"\tTotal cloud tasks: {self.cloud_tasks}")
        print(f"\tTotal local execution tasks: {self.local_execution}")
        print(f"\tTotal completed tasks: {self.completed_tasks}")
        print(f"\tTotal tasks: {self.total_tasks}")
        if self.total_tasks != 0:
            print(f"\tMigration ratio: {'{:.3f}'.format(self.migrations_count * 100 / self.total_tasks)}%")
            print(f"\tDeadline miss ratio: {'{:.3f}'.format(self.deadline_misses * 100 / self.total_tasks)}%")
            if self.deadline_misses:
                print(
                    f"\tNo Resource found by deadline miss ratio: "
                    f"{'{:.3f}'.format(self.no_resource_found * 100 / self.deadline_misses)}%"
                )
