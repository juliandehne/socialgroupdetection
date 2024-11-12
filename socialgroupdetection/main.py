import os
import json
import requests
from .system_prompt import system_prompt


class SGA:
    gwdg_model = "mistral-large-instruct"
    chatgpt_model = "gpt-4"
    gwdg_url = "https://chat-ai.academiccloud.de/v1/chat/completions"
    # gwdg_url = "https://chat-ai.academiccloud.de/v1/models"
    chatgpt_url = "https://api.openai.com/v1/chat/completions"
    max_tokens = 150
    temperature = 0.5

    def __init__(self, bearer_key=None, gwdg_server=True, chatgpt_server=False):
        if bearer_key is None:
            # Load bearer key from environment variable
            bearer_key = os.getenv("BEARER_KEY")
            if bearer_key is None:
                raise ValueError(
                    "Bearer key is required. Set it in the environment or pass it as an argument,"
                    " i.e. SGA(bearer_key=<your key>).")

        self.bearer_key = bearer_key

        # Set server configurations based on the selected server
        if gwdg_server:
            self.url = self.gwdg_url
            self.model = self.gwdg_model
        elif chatgpt_server:
            self.url = self.chatgpt_url
            self.model = self.chatgpt_model
        else:
            raise ValueError("Specify either GWDG server or ChatGPT server.")

    def get_completion(self, prompt):
        """
        Sends a prompt to the configured server and returns the response.

        Parameters:
        - prompt (str): The prompt to send to the model.
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): The temperature for response variability.

        Returns:
        - dict: The JSON response from the API.
        """
        headers = {
            "Authorization": f"Bearer {self.bearer_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "top_p": 0.15,
        }
        # Make the POST request
        response = requests.post(self.url, headers=headers, json=data)
        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            # Extract and return only the text response
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            return None  # Optionally, return the error response
