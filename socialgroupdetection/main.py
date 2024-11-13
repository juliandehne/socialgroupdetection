import os
import json

from tqdm import tqdm
import pandas as pd

import requests

from .system_prompt import system_prompt
from .filtering import *
from .dictionary import no_social_groups as blacklist_words


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

    def get_social_groups(self, texts_or_text, include_implicit=False, embedding_based_filtering=False,
                          filter_type="linear", as_dataframe=False):
        """
         Sends a prompt to the configured server and returns the response.

         Parameters:
         - prompt (str or list[str]): The prompt to send to the model.
         - max_tokens (int): The maximum number of tokens to generate.
         - temperature (float): The temperature for response variability.
         - embedding_based_filtering (Bool): Whether the results should be filtered based on word embeddings geometrically
         - filter: can be "linear", "svm", or "svm2"
         - as_dataframe: if True returns data frame with all the combinations generated

         Returns:
         - list[dict]: The JSON response from the API.
         """
        # settings:
        tqdm.pandas()

        if isinstance(texts_or_text, str):
            return self.__get_social_groups([texts_or_text], include_implicit, embedding_based_filtering, filter_type,
                                            as_dataframe)
        else:
            return self.__get_social_groups(texts_or_text, include_implicit, embedding_based_filtering, filter_type,
                                            as_dataframe)

    def __get_social_groups(self, texts, embedding_based_filtering, include_implicit, filter_type, as_dataframe):
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
            result = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Remove markdown formatting (backticks and "json" label)
            result = result.replace('```', "").replace('json', '').replace("\n", "")
            result = json.loads(result)
            results.append(result)

        results = pd.DataFrame(results, columns=["explizit", "implicit", "sonstige"])

        # adds the implicit groups found by mistral to the process
        if include_implicit:
            results["explicit"] = results["explizit"] + results["implicit"]
        else:
            results["explicit"] = results["explizit"]

        # perform filtering
        if embedding_based_filtering or as_dataframe:

            results["new_embeddings"] = convert_terms_to_embeddings(results["explicit"], use_cls_token=True)

            magic_white_list = ['soldiers', 'farmers', 'self-employed', 'care personnel', 'entrepreneurs',
                                'university graduates', 'first-time voters', 'parents', 'women',
                                'people with lower education', 'Muslims', 'business founders']
            white_list_centroids = convert_terms_to_embeddings(magic_white_list, use_cls_token=True)

            if filter_type == "linear" or as_dataframe:
                # Classify using the linear method
                def filter_non_groups_linear(new_embeddings, list_of_labels):
                    try:
                        classifications_linear = classify_with_linear(new_embeddings, white_list_centroids)
                        filtered_list_linear = [item for item, keep in zip(list_of_labels, classifications_linear) if
                                                keep == 1]
                        return filtered_list_linear
                    except Exception as ex:
                        # print(ex)
                        return list_of_labels

                results["linear"] = results.progress_apply(
                    lambda row: filter_non_groups_linear(row["new_embeddings"], row["explicit"]), axis=1)

            if filter_type == "svm" or as_dataframe:
                magic_words_svm = ['cleaning personnel', 'researchers', 'university graduates', 'urban population',
                                   'jobless', 'pensioners', 'care personnel', 'women', 'farmers', 'employers',
                                   'first-time voters', 'people with lower education']

                white_list_centroids_svm = convert_terms_to_embeddings(magic_words_svm, use_cls_token=True)
                oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.2)
                oc_svm.fit(white_list_centroids_svm)

                def filter_non_groups_svm(new_embeddings, list_of_labels):
                    try:
                        classification_svm = classify_with_svm(new_embeddings, white_list_centroids_svm, oc_svm=oc_svm)
                        filtered_list_svm = [item for item, keep in zip(list_of_labels, classification_svm) if
                                             keep == 1]
                        return filtered_list_svm
                    except Exception as ex:
                        # print(ex)
                        return list_of_labels

                results["svm"] = results.progress_apply(
                    lambda row: filter_non_groups_svm(row["new_embeddings"], row["explicit"]), axis=1)

            if filter_type == "svm2" or as_dataframe:
                black_list_centroids = convert_terms_to_embeddings(blacklist_words, use_cls_token=True)

                def filter_non_groups_svm(new_embeddings, list_of_labels):
                    try:
                        # Combine white list and black list embeddings to form the training data
                        X_train = np.vstack([white_list_centroids, black_list_centroids])

                        # Create labels: 1 for white list (positive class), -1 for black list (negative class)
                        y_train = np.hstack(
                            [np.ones(len(white_list_centroids)), -1 * np.ones(len(black_list_centroids))])

                        # Train a two-class SVM
                        two_class_svm = SVC(kernel='linear')  # You can choose other kernels like 'rbf' if needed
                        two_class_svm.fit(X_train, y_train)

                        classifications_svm2 = classify_with_two_class_svm(new_embeddings, white_list_centroids,
                                                                           black_list_centroids, two_class_svm)
                        filtered_list_svm2 = [item for item, keep in zip(list_of_labels, classifications_svm2) if
                                              keep == 1]
                        return filtered_list_svm2
                    except Exception as ex:
                        # print(ex)
                        return list_of_labels

                results["svm2"] = results.progress_apply(
                    lambda row: filter_non_groups_svm(row["new_embeddings"], row["explicit"]), axis=1)

        if as_dataframe:
            return results
        if not embedding_based_filtering:
            return results["explicit"]
        else:
            return results[filter_type]
