import json
import os
from typing import Dict, Any

class FinancialAnalystMemory:
    def __init__(self, path):
        self.path = path
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {'tasks': {}, 'results': {}}

    def _save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f)

    def add_task(self, task_data) -> str:
        task_id = str(len(self.data['tasks']) + 1)
        self.data['tasks'][task_id] = {'data': task_data, 'status': 'pending'}
        self._save()
        return task_id

    def complete_task(self, task_id, result):
        self.data['tasks'][task_id]['status'] = 'completed'
        self.data['results'][task_id] = result
        self._save()

    def get_task_report(self, task_id):
        task = self.data['tasks'].get(task_id)
        result = self.data['results'].get(task_id)
        return {'task': task, 'result': result}

    def list_tasks(self):
        return self.data['tasks']
