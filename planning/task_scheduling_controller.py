# planning/task_scheduling_controller.py
class TaskSchedulingController:
    """
    Dynamically schedules and prioritizes tasks.
    """
    def __init__(self):
        self.task_queue = []

    def add_task(self, task, priority=1.0):
        self.task_queue.append((priority, task))
        self.task_queue.sort(key=lambda x: -x[0])

    def pick_next_task(self):
        if self.task_queue:
            return self.task_queue.pop(0)[1]
        return None

