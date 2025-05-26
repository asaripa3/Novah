def run_chat_session(
    profile,
    query_agent,
    memory_agent,
    context_agent,
    planner_agent,
    responder_agent,
    sanitizer_agent,
    psychiatrist_agent,
    save_profile,
    input_message=None,
    response_queue=None,
    chat_history=None
):
    """
    Run a chat session with the given agents.
    
    Args:
        profile: User profile containing preferences and settings
        query_agent: Agent for parsing user queries
        memory_agent: Agent for retrieving relevant memories
        context_agent: Agent for filtering context
        planner_agent: Agent for planning responses
        responder_agent: Agent for generating responses
        sanitizer_agent: Agent for sanitizing responses
        psychiatrist_agent: Agent for emotional analysis and memory processing
        save_profile: Function to save profile updates
        input_message: Optional message for web interface
        response_queue: Optional queue for web interface communication
        chat_history: List of previous chat messages
    """
    try:
        # Initialize chat history if not provided
        if chat_history is None:
            chat_history = []

        # Process the input message
        if input_message:
            # Parse the query
            query_result = query_agent.parse_query(input_message)
            
            # Process with psychiatrist agent
            psychiatrist_result = psychiatrist_agent.process_input(input_message, query_result)
            
            # Update profile with psychiatrist results
            if "profile_updates" in psychiatrist_result:
                profile.update(psychiatrist_result["profile_updates"])
                save_profile(profile, "../data/yahya_profile.jsonl")

            # Retrieve relevant memories
            memories = memory_agent.retrieve(
                query_keywords=query_result["query_keywords"],
                emotion=query_result["emotion"]
            )

            # Filter context
            filtered_context = context_agent.filter(memories=memories, query_emotion=query_result["emotion"])
            
            # Get top memory for response
            top_memory = filtered_context[0]['text'] if filtered_context else "No specific memory was recalled for this question."
            
            # Build prompt with chat history
            dialogue_context = "\n".join(chat_history[-12:]) if chat_history else ""
            prompt = planner_agent.build_prompt(input_message, top_memory, profile, dialogue_context)
            
            # Generate response
            raw_response = responder_agent.get_response(prompt)
            
            # Sanitize response
            final_response = sanitizer_agent.sanitize(raw_response)
            
            # Update chat history
            chat_history.append(f"User: {input_message}")
            chat_history.append(f"Assistant: {final_response}")
            
            # If using web interface, put response in queue
            if response_queue:
                response_queue.put(final_response)
                return
            
            return final_response
            
        else:
            # Interactive console mode
            print("Welcome to NovahSpeaks! Type 'quit' to exit.")
            while True:
                user_input = input("You: ").strip()

                if user_input.lower() == 'quit':
                    break

                # Parse the query
                query_result = query_agent.parse_query(user_input)
                
                # Process with psychiatrist agent
                psychiatrist_result = psychiatrist_agent.process_input(user_input, query_result)
                
                # Update profile with psychiatrist results
                if "profile_updates" in psychiatrist_result:
                    profile.update(psychiatrist_result["profile_updates"])
                    save_profile(profile, "../data/yahya_profile.jsonl")
                
                # Retrieve relevant memories
                memories = memory_agent.retrieve(
                    query_keywords=query_result["query_keywords"],
                    emotion=query_result["emotion"]
                )
                
                # Filter context
                filtered_context = context_agent.filter(memories=memories, query_emotion=query_result["emotion"])

                # Get top memory for response
                top_memory = filtered_context[0]['text'] if filtered_context else "No specific memory was recalled for this question."
                
                # Build prompt with chat history
                dialogue_context = "\n".join(chat_history[-12:]) if chat_history else ""
                prompt = planner_agent.build_prompt(user_input, top_memory, profile, dialogue_context)
                
                # Generate response
                raw_response = responder_agent.get_response(prompt)
                
                # Sanitize response
                final_response = sanitizer_agent.sanitize(raw_response)

                # Update chat history
                chat_history.append(f"User: {user_input}")
                chat_history.append(f"Assistant: {final_response}")
                
                print(f"NovahSpeaks: {final_response}")
                
    except Exception as e:
        error_message = f"Error in chat session: {str(e)}"
        if response_queue:
            response_queue.put(error_message)
        else:
            print(error_message)
        return error_message