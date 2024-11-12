import os
import json

import pandas as pd
import requests

from .embeddings import convert_terms_to_embeddings
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

    def get_social_groups(self, texts_or_text, embedding_based_filtering=False):
        """
         Sends a prompt to the configured server and returns the response.

         Parameters:
         - prompt (str or list[str]): The prompt to send to the model.
         - max_tokens (int): The maximum number of tokens to generate.
         - temperature (float): The temperature for response variability.
         - embedding_based_filtering (Bool): Whether the results should be filtered based on word embeddings geometrically

         Returns:
         - list[dict]: The JSON response from the API.
         """
        if isinstance(texts_or_text, str):
            return self.__get_social_groups([texts_or_text], embedding_based_filtering)
        else:
            return self.__get_social_groups(texts_or_text, embedding_based_filtering)

    def __get_social_groups(self, texts, embedding_based_filtering):
        headers = {
            "Authorization": f"Bearer {self.bearer_key}",
            "Content-Type": "application/json"
        }

        results = []
        for text in texts:
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": self.temperature,
                "top_p": 0.15,
            }
            # Make the POST request
            response = requests.post(self.url, headers=headers, json=data)
            # Check if the request was successful
            if response.status_code != 200:
                return None  # Optionally, return the error response

            response_data = response.json()
            # Extract and return only the text response
            result  =  response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Remove markdown formatting (backticks and "json" label)
            result = result.replace('```', "").replace('json', '').replace("\n","")
            result = json.loads(result)
            results.append(result)


        results = pd.DataFrame(results, columns=["explizit", "implicit", "sonstige"])

        if embedding_based_filtering:
            magic_white_list = ['soldiers', 'farmers', 'self-employed', 'care personnel', 'entrepreneurs', 'university graduates', 'first-time voters', 'parents', 'women', 'people with lower education', 'Muslims', 'business founders']
            magic_words_svm = ['cleaning personnel', 'researchers', 'university graduates', 'urban population', 'jobless', 'pensioners', 'care personnel', 'women', 'farmers', 'employers', 'first-time voters', 'people with lower education']

            white_list_centroids = convert_terms_to_embeddings(magic_white_list, use_cls_token=True)
            white_list_centroids_svm = convert_terms_to_embeddings(magic_words_svm, use_cls_token=True)

        return results
