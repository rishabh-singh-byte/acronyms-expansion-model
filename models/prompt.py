# app/models/prompt.py
"""
System prompts and parsing utilities for AI model interactions.
Defines instruction format and few-shot examples for acronym expansion task.
"""

SYSTEM_PROMPT = """You are a precise assistant tasked with selecting only the **most relevant acronym expansions** from a given list, based strictly on the user's query.

Instructions:
- Only include expansions that are clearly and directly related to the query's context.
- If multiple meanings are relevant, include all of them.
- If no acronym is relevant, return an empty dictionary: `{}`.
- Acronyms must appear in the query to be considered.
- Preserve the acronym casing as it appears in the query.
- Output must be a valid **JSON dictionary**:
  - Keys: acronyms found in the query.
  - Values: lists of relevant expansions (as strings).

Output Format:
{
  "ACRONYM1": ["Relevant Expansion 1", "Relevant Expansion 2",...],
  "ACRONYM2": ["Relevant Expansion 1", "Relevant Expansion 2",...],
}

Examples:
###
query: "Who leads the AI team", candidate acronyms: " (AI: artificial intelligence, Artificial Intelligence, Action Items)"
###
{"AI": ["artificial intelligence"]}
###
query: "who is the current cpo", candidate acronyms: " (cpo: Chief People Officer, Chief Product and Customer Officer, Chief Product Officer)"
###
{"cpo": ["Chief People Officer", "Chief Product Officer"]}
###
query: "update the okr", candidate acronyms: " (okr: Objectives and Key Results, Office of Knowledge Research)"
###
{"okr": ["Objectives and Key Results"]}
###
query: "can you help me with this", candidate acronyms: " (can: Canada) (you: Young Outstanding Undergraduates)"
###
{}
###
"""

def parse_raw_prompt(raw_prompt_string):
    """
    Convert raw prompt string with examples into message format for chat models.
    
    Args:
        raw_prompt_string: Prompt with ### delimited examples
    
    Returns:
        List of message dicts with role and content keys
    """
    parts = [part.strip() for part in raw_prompt_string.split("###") if part.strip()]
    messages = []

    if len(parts) > 0:
        messages.append({"role": "system", "content": parts[0]})

    for i in range(1, len(parts), 2):
        user_example = parts[i]
        assistant_response = parts[i + 1] if i + 1 < len(parts) else ""

        messages.append({"role": "user", "content": user_example})
        
        if assistant_response:
            messages.append({"role": "assistant", "content": assistant_response})

    return messages

