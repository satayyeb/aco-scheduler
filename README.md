# 🚗 Multi-Zone Task Offloading with Priority and Preemption Simulation

This project simulates a task offloading system for connected vehicles in a multi-zone environment consisting of **Cloud**, **Fog**, **Local**, and a **60%-capacity-limited special region**. It models various types of tasks and makes offloading decisions based on task criticality, zone capacities, and a zone manager hierarchy. The goal is to achieve efficient task distribution, real-time constraints, and minimal deadline misses under dynamic workloads.

---

## 📌 Key Features

- 🧠 **Task Prioritization**: Tasks are classified into three categories:
  - `crucial` — must mostly be offloaded to the 60% zone
  - `high_critical` — can only go to Cloud, Fog, or Local
  - `low_critical` — can go anywhere, including the 60% zone if not full

- 🧪 **Task Generation**:
  - `crucial` tasks: generated **periodically**
  - `high_critical` and `low_critical` tasks: generated using **Poisson** distribution

- 📤 **Offloading Strategy**:
  - Each `Zone Manager` selects the best local offloading option using its algorithm.
  - Among all zones, the **best global offloading decision** is made centrally.
  - If the 60% zone is full but needs to accept new crucial tasks, **preemption** is triggered, and existing `low_critical` tasks are re-evaluated and re-offloaded.

- ⚙️ **Simulation Environment**:
  - Task migration is disabled for simplicity.
  - Includes metrics such as deadline miss rate, preemption count, task distributions, etc.

---
