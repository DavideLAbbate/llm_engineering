import logging
import json
from chatbot_state import task_state_manager
from chatbot_utils import get_claude_response, get_openai_response

logger = logging.getLogger(__name__)
MAX_CORRECTION_RETRIES = 3
MIN_AVG_SCORE = 0.6

def verify_single_part(part_number, history, flowchart):
    """Uses Claude to verify a single response part and returns the score JSON."""
    logger.info(f"VERIFIER: Verifying part {part_number}...")
    
    response_data = task_state_manager.get_response_part(part_number)
    if not response_data:
        logger.error(f"VERIFIER: No response data found for part {part_number}.")
        return None

    current_desc = f"Content for part {part_number}."
    prev_desc = f"Content for part {part_number - 1}." if part_number > 1 else None
    next_desc = f"Content for part {part_number + 1}." if part_number < task_state_manager.analysis_json['numberOfResponse'] else None

    with open("prompts/multi_part_verifier_prompt_strict.txt", "r", encoding="utf-8") as f:
        verifier_prompt_template = f.read()

    system_prompt = verifier_prompt_template
    user_prompt = f"""
    {{
        "history": {json.dumps(history)},
        "flowchart": {json.dumps(flowchart)},
        "response": {json.dumps(response_data['content'])},
        "contextPart": {part_number},
        "contextualDescription": {json.dumps(current_desc)},
        "previousContextualDescription": {json.dumps(prev_desc)},
        "nextContextualDescription": {json.dumps(next_desc)}
    }}
    """
    
    raw_response = get_claude_response(system_prompt, user_prompt, is_json=True)
    
    try:
        score_data = json.loads(raw_response)
        task_state_manager.add_score_data(part_number, score_data)
        logger.info(f"VERIFIER: Score for part {part_number}: {score_data}")
        return score_data
    except json.JSONDecodeError:
        logger.error(f"VERIFIER: Failed to decode JSON from verification response for part {part_number}.")
        return None

def correct_failed_part(part_number, history, flowchart):
    """Attempts to correct a part that failed verification."""
    logger.warning(f"VERIFIER: Attempting to correct failed part {part_number}.")
    part_data = task_state_manager.get_response_part(part_number)
    if not part_data or not part_data.get('score_data'):
        return False

    score_data = part_data['score_data']
    critical_issues = score_data.get('criticalIssues', [])
    
    system_prompt = (
        "You are a correction AI. You will receive a piece of content that failed a quality check, along with the critical issues found. "
        "Your task is to rewrite the content to resolve these issues, adhering to the original plan. "
        "Provide only the corrected, full text for this part."
    )
    
    issues_block = "- " + "\n- ".join(critical_issues) if critical_issues else "(none)"
    
    user_prompt = (
        f"The following content for Part {part_number} failed verification.\n\n"
        f"**Original Content:**\n{part_data['content']}\n\n"
        f"**Critical Issues to Fix:**\n{issues_block}\n\n"
        f"**Overall Plan (Flowchart):**\n{flowchart}\n\n"
        "Please provide the corrected version of the content for this part."
    )

    logger.info("VERIFIER: critical issues block:\n%s", issues_block)

    corrected_content = get_openai_response(system_prompt, user_prompt, history)
    task_state_manager.update_response_part(part_number, corrected_content)
    logger.info(f"VERIFIER: Part {part_number} has been re-generated.")
    return True


def run_advanced_verification(history, flowchart):
    """Manages the verification and correction loop for all response parts."""
    total_parts = task_state_manager.analysis_json['numberOfResponse']

    for attempt in range(MAX_CORRECTION_RETRIES + 1):
        yield f"Ciclo di verifica ({attempt + 1}/{MAX_CORRECTION_RETRIES})..."
        
        failed_parts = []
        for part_num in range(1, total_parts + 1):
            yield f"Verifica parte {part_num} di {total_parts}..."
            score_data = verify_single_part(part_num, history, flowchart)
            
            if not score_data:
                failed_parts.append(part_num)
                continue

            avg_score = (score_data.get('continuityScore', 0) + 
                         score_data.get('semanticCoherenceScore', 0) + 
                         score_data.get('alignmentWithStepScore', 0)) / 3
            
            if not score_data.get('isValid', False) or avg_score < MIN_AVG_SCORE:
                logger.warning(f"VERIFIER: Part {part_num} failed verification. Avg score: {avg_score:.2f}. Issues: {score_data.get('criticalIssues')}")
                failed_parts.append(part_num)

        if not failed_parts:
            logger.info("VERIFIER: All parts passed verification.")
            yield "Verifica completata con successo."
            return True

        if attempt >= MAX_CORRECTION_RETRIES:
            logger.error("VERIFIER: Max correction retries reached. Aborting.")
            yield "Errore: impossibile correggere la risposta dopo i tentativi massimi."
            return False

        yield f"Correzione di {len(failed_parts)} parti fallite..."
        for part_num in failed_parts:
            correct_failed_part(part_num, history, flowchart)
            
    return False
