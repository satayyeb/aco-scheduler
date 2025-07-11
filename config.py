class Config:
    CHUNK_SIZE = 1200

    class SimulatorConfig:
        SIMULATION_DURATION = 100
        BANDWIDTH = 150

    class CloudConfig:
        DEFAULT_X = 6000
        DEFAULT_Y = 1500
        DEFAULT_RADIUS = 10000
        CLOUD_BANDWIDTH = 60
        MAX_TASK_QUEUE_LEN = 2000
        DEFAULT_COMPUTATION_POWER = 1000
        POWER_LIMIT = 0.99

    class FixedFogNodeConfig:
        MAX_TASK_QUEUE_LEN = 400
        DEFAULT_COMPUTATION_POWER = 500
        POWER_LIMIT = 0.9

    class MobileFogNodeConfig:
        DEFAULT_RADIUS = 150
        MAX_TASK_QUEUE_LEN = 150
        DEFAULT_COMPUTATION_POWER = 200
        POWER_LIMIT = 0.6

    class UserNodeConfig:
        MAX_TASK_QUEUE_LEN = 10
        DEFAULT_COMPUTATION_POWER = 20
        LOCAL_OFFLOAD_POWER_OVERHEAD = 1
        LOCAL_EXECUTE_TIME_OVERHEAD = 1
        POWER_LIMIT = 0.4

    class ZoneManagerConfig:
        ALGORITHM_RANDOM = "Random"
        ALGORITHM_HEURISTIC = "Heuristic"
        ALGORITHM_HRL = "HRL"
        ALGORITHM_ONLY_CLOUD = "Only Cloud"
        ALGORITHM_ONLY_FOG = "Only Fog"
        ALGORITHM_DEEP_RL = "DeepRL"
        ALGORITHM_ACO = "ACO"

        DEFAULT_ALGORITHM = ALGORITHM_ACO

    class FinalDeciderMethod:
        FIRST_CHOICE = "First Choice"
        RANDOM_CHOICE = "Random Choice"
        MIN_DISTANCE = "Min Distance"
        ACO_BASED = "ACO Based"

        DEFAULT_METHOD = FIRST_CHOICE

    class RandomZoneManagerConfig:
        OFFLOAD_CHANCE: float = 0.5

    class TaskConfig:
        PACKET_COST_PER_METER = 0.001

        TASK_COST_PER_METER = 0.005

        MIGRATION_OVERHEAD = 0.01
        CLOUD_PROCESSING_OVERHEAD = 0.5


