from vertexai.preview.generative_models import GenerativeModel



class ChatLLM:
    def __init__(self):
        self.model = GenerativeModel("gemini-2.0-flash-001")

        # Manually injecting system prompt as first message
        system_instruction = "Act as a chatbot and do not give answers in more than 50 words."

        self.chat = self.model.start_chat(history=[])
        self.chat.send_message(system_instruction)

    def stream_message(self, user_input: str):
        try:
            response_stream = self.chat.send_message(user_input, stream=True)
            for chunk in response_stream:
                yield chunk.text
        except Exception as e:
            yield f"Error: {str(e)}"