let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let stream;

// Function to stop recording
function stopRecording() {
    if (isRecording) {
        console.log('Stopping recording...');
        mediaRecorder.stop();
        stream.getTracks().forEach(track => track.stop());
        isRecording = false;
        document.getElementById('mic-button').classList.remove('text-red-500');
        document.getElementById('recording-overlay').classList.add('hidden');
    }
}

// Start recording when mic button is clicked
document.getElementById('mic-button').addEventListener('click', async () => {
    if (!isRecording) {
        console.log('Starting recording...');
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            console.log('Microphone access granted');
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm',
                audioBitsPerSecond: 128000
            });
            
            audioChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    console.log('Received audio chunk:', event.data.size, 'bytes');
                    audioChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = async () => {
                console.log('Recording stopped, processing audio...');
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                console.log('Audio blob created:', audioBlob.size, 'bytes');
                const formData = new FormData();
                formData.append('audio', audioBlob, 'audio.webm');
                
                try {
                    console.log('Sending audio for transcription...');
                    const response = await fetch('/api/transcribe', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    console.log('Transcription response:', data);
                    
                    if (data.text) {
                        console.log('Setting transcribed text:', data.text);
                        const messageInput = document.getElementById('message-input');
                        messageInput.value = data.text;
                        
                        console.log('Submitting chat form...');
                        document.getElementById('chat-form').dispatchEvent(new Event('submit'));
                    } else {
                        console.error('No text in transcription response');
                    }
                } catch (error) {
                    console.error('Error in transcription process:', error);
                }
            };
            
            mediaRecorder.start();
            isRecording = true;
            document.getElementById('mic-button').classList.add('text-red-500');
            document.getElementById('recording-overlay').classList.remove('hidden');
            console.log('Recording started');
        } catch (error) {
            console.error('Error accessing microphone:', error);
        }
    }
});

// Stop recording when stop button is clicked
document.getElementById('stop-recording').addEventListener('click', stopRecording); 