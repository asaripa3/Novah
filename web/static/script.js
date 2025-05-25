document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    const speechAudio = document.getElementById('speech-audio');

    // Focus input on load
    messageInput.focus();

    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
        
        // Add avatar and name header
        const header = document.createElement('div');
        header.className = 'flex items-center space-x-2 mb-1';
        
        if (isUser) {
            // User message header
            header.innerHTML = `
                <div class="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
                    <span class="text-white text-sm">${window.userInfo.emoji}</span>
                </div>
                <span class="text-sm text-gray-500">${window.userInfo.name}</span>
            `;
        } else {
            // Assistant message header
            header.innerHTML = `
                <div class="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center">
                    <span class="text-indigo-600 text-sm font-semibold">N</span>
                </div>
                <span class="text-sm text-gray-500">NovahSpeaks</span>
            `;
        }
        
        messageDiv.appendChild(header);
        
        const messageContent = document.createElement('p');
        messageContent.textContent = message;
        messageDiv.appendChild(messageContent);
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function addTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        chatMessages.appendChild(indicator);
        scrollToBottom();
        return indicator;
    }

    function removeTypingIndicator(indicator) {
        if (indicator && indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
        }
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Handle form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = messageInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addMessage(message, true);
        messageInput.value = '';
        messageInput.focus();

        // Show typing indicator
        const typingIndicator = addTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message }),
            });

            const data = await response.json();

            // Remove typing indicator
            removeTypingIndicator(typingIndicator);

            if (data.status === 'success') {
                addMessage(data.response);
                
                // Play audio if available
                if (data.hasAudio) {
                    console.log('Audio available, attempting to play...');
                    // Add timestamp to prevent caching
                    const timestamp = new Date().getTime();
                    const audioUrl = `/api/speech?t=${timestamp}`;
                    console.log('Loading audio from:', audioUrl);
                    
                    // Reset audio element
                    speechAudio.pause();
                    speechAudio.currentTime = 0;
                    
                    // Set up audio event listeners
                    speechAudio.oncanplaythrough = () => {
                        console.log('Audio can play through, starting playback...');
                        speechAudio.play().catch(error => {
                            console.error('Error playing audio:', error);
                        });
                    };
                    
                    speechAudio.onerror = (e) => {
                        console.error('Audio loading error:', e);
                    };
                    
                    speechAudio.onended = () => {
                        console.log('Audio playback completed');
                    };
                    
                    // Load the audio
                    speechAudio.src = audioUrl;
                } else {
                    console.log('No audio available for this response');
                }
            } else {
                addMessage('Sorry, I encountered an error. Please try again.');
                console.error('Error:', data.error);
            }
        } catch (error) {
            removeTypingIndicator(typingIndicator);
            addMessage('Sorry, I encountered an error. Please try again.');
            console.error('Error:', error);
        }
    });

    // Handle input focus
    messageInput.addEventListener('focus', () => {
        messageInput.parentElement.classList.add('ring-2', 'ring-indigo-200');
    });

    messageInput.addEventListener('blur', () => {
        messageInput.parentElement.classList.remove('ring-2', 'ring-indigo-200');
    });

    // Handle enter key
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });
}); 