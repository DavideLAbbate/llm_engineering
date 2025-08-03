import logging
from state_manager import task_state_manager
from chatbot_utils import get_openai_response, get_claude_response # We'll create this file

logger = logging.getLogger(__name__)

def generate_response_part(part_number, total_parts, user_prompt, history, flowchart, contextual_recap):
    """Generates a single part of a multi-part response."""
    logger.info(f"ORCHESTRATOR: Generating response for part {part_number}/{total_parts}...")

    system_prompt = (
        "You are an AI assistant executing one part of a larger, structured task. "
        "You must generate ONLY the content for the current part, ensuring it is coherent and flows logically from the previous context."
    )

    previous_responses_content = ""
    if part_number > 1:
        prev_parts = []
        for i in range(1, part_number):
            part_data = task_state_manager.get_response_part(i)
            if part_data:
                prev_parts.append(f"--- START OF PART {i} ---\n{part_data['content']}\n--- END OF PART {i} ---")
        previous_responses_content = "\n\n".join(prev_parts)


    user_context_prompt = (
        f"This is part **{part_number}** of a **{total_parts}-part** task.\n\n"
        f"**Original User Request:**\n{user_prompt}\n\n"
        f"**Overall Plan (Flowchart):**\n{flowchart}\n\n"
    )

    if contextual_recap:
        user_context_prompt += f"**Contextual Recap of Previous Work:**\n{contextual_recap}\n\n"
    
    if previous_responses_content:
         user_context_prompt += f"**Content of Previous Parts:**\n{previous_responses_content}\n\n"

    user_context_prompt += f"Your task is to generate the content for **PART {part_number} ONLY**. Do not repeat the plan or previous parts. Focus solely on fulfilling the requirements for this specific part as outlined in the flowchart."

    # Using a helper for the actual API call
    response_content = get_openai_response(system_prompt, user_context_prompt, history)
    return response_content


def run_multi_response_pipeline(user_prompt, history):
    """Manages the entire pipeline for generating multi-part responses."""
    analysis = task_state_manager.analysis_json
    start_part = analysis['currentContextPart']
    total_parts = analysis['numberOfResponse']
    
    # Loop to generate all required parts
    for part_num in range(start_part, total_parts + 1):
        yield f"Generazione parte {part_num} di {total_parts}..."
        
        content = generate_response_part(
            part_number=part_num,
            total_parts=total_parts,
            user_prompt=user_prompt,
            history=history,
            flowchart=task_state_manager.flowchart,
            contextual_recap=analysis.get('contextualRecap')
        )
        task_state_manager.add_response_part(part_num, content)

    logger.info("ORCHESTRATOR: All response parts generated.")
