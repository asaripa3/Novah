def process_input(
    user_input,
    profile,
    chat_history,
    query_agent,
    memory_agent,
    context_agent,
    planner_agent,
    responder_agent,
    sanitizer_agent
):
    parsed = query_agent.parse_query(user_input)

    retrieved = memory_agent.retrieve(
        query_keywords=parsed["query_keywords"],
        emotion=parsed["emotion"]
    )

    filtered = context_agent.filter(memories=retrieved, query_emotion=parsed["emotion"])

    top_memory = filtered[0]['text'] if filtered else "No specific memory was recalled for this question."
    dialogue_context = "\n".join(chat_history[-12:])
    prompt = planner_agent.build_prompt(user_input, top_memory, profile, dialogue_context)
    response = responder_agent.get_response(prompt)
    response = sanitizer_agent.sanitize(response)

    return response, parsed, retrieved, filtered, top_memory
