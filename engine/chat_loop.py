def run_chat_session(
    profile,
    query_agent,
    memory_agent,
    context_agent,
    planner_agent,
    responder_agent,
    sanitizer_agent,
    save_profile_func
):
    import nltk
    from nltk.tokenize import word_tokenize
    from utils.text_utils import normalize_list

    chat_history = []

    while True:
        user_input = input(f"\n{profile['name']}: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Step 1: Parse query
        parsed = query_agent.parse_query(user_input)
        print("\n[Parsed Query]")
        print(parsed)

        # Step 2: Retrieve relevant memories
        retrieved = memory_agent.retrieve(
            query_keywords=parsed["query_keywords"],
            emotion=parsed["emotion"]
        )

        # Update vocabulary from user input
        tokens = word_tokenize(user_input)
        pos_tags = nltk.pos_tag(tokens)
        filtered_tokens = [word for word, pos in pos_tags if pos.startswith('NN') or pos.startswith('VB')]
        new_words = normalize_list(filtered_tokens)
        updated = False
        for word in new_words:
            if word not in profile["known_vocabulary"]:
                profile["known_vocabulary"].append(word)
                updated = True

        # Update trigger_words if emotion suggests sensitivity
        if parsed["emotion"] in {"anxious", "sad", "angry"}:
            for word in new_words:
                if word not in profile.get("trigger_words", []):
                    profile["trigger_words"].append(word)
                    updated = True

        # Update preferred_topics based on common interest words
        frequent_topic_keywords = {"books", "games", "videos", "animals", "trains"}
        for word in new_words:
            if word in frequent_topic_keywords and word not in profile.get("preferred_topics", []):
                profile["preferred_topics"].append(word)
                updated = True

        if updated:
            save_profile_func(profile, "data/yahya_profile.jsonl")
            context_agent.update_known_vocabulary(profile.get("known_vocabulary", []))

        print("\n[Memory Scoring Debug Before Context Filter]")
        for mem in retrieved:
            print(f"- {mem['text']} (Emotion: {mem['emotion']}, Tags: {mem['tags']})")

        filtered = context_agent.filter(memories=retrieved, query_emotion=parsed["emotion"])

        print("\n[Filtered Memories After Context Filter]")
        for mem in filtered:
            print(f"- {mem['text']} (Importance: {mem.get('importance_score', 0)}, Vocab: {mem.get('vocabulary', [])})")

        top_memory = filtered[0]['text'] if filtered else "No specific memory was recalled for this question."
        dialogue_context = "\n".join(chat_history[-12:])
        prompt = planner_agent.build_prompt(user_input, top_memory, profile, dialogue_context)
        print("\n[Generated Prompt for LLM]")
        print(prompt)
        response = responder_agent.get_response(prompt)
        response = sanitizer_agent.sanitize(response)

        print("\nNovah:")
        print(response)

        chat_history.append(f"{profile['name']}: {user_input}")
        chat_history.append(f"Bot: {response}")