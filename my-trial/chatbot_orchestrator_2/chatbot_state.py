import logging

logger = logging.getLogger(__name__)

class TaskStateManager:
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the state for a new task."""
        logger.info("STATE: Resetting task state.")
        self.is_active = False
        self.analysis_json = {}
        self.flowchart = ""
        self.response_parts = {} # {1: {"content": "...", "score_data": {...}}, 2: ...}

    def start_task(self, analysis_json, flowchart):
        """Initializes the state for a new multi-part task."""
        self.reset()
        self.is_active = True
        self.analysis_json = analysis_json
        self.flowchart = flowchart
        logger.info(f"STATE: New multi-part task started. Analysis: {analysis_json}")

    def add_response_part(self, part_number, content):
        """Adds a generated response part to the state."""
        self.response_parts[part_number] = {"content": content, "score_data": None}
        logger.info(f"STATE: Added content for part {part_number}.")

    def update_response_part(self, part_number, new_content):
        """Updates the content of a response part after correction."""
        if part_number in self.response_parts:
            self.response_parts[part_number]['content'] = new_content
            self.response_parts[part_number]['score_data'] = None # Reset score after correction
            logger.info(f"STATE: Updated content for part {part_number}.")

    def add_score_data(self, part_number, score_data):
        """Adds verification score data to a response part."""
        if part_number in self.response_parts:
            self.response_parts[part_number]['score_data'] = score_data
            logger.info(f"STATE: Added score for part {part_number}.")

    def get_response_part(self, part_number):
        """Retrieves a specific response part."""
        return self.response_parts.get(part_number)

    def get_all_responses(self):
        """Retrieves all response parts in order."""
        return [self.response_parts[i] for i in sorted(self.response_parts.keys())]

# Singleton instance of the state manager
task_state_manager = TaskStateManager()
