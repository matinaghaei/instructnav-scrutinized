from openai import OpenAI
import retry
import numpy as np

V2_SYSTEM_PROMPT_NEGATIVE = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should not waste time exploring. For example if we are in a house and looking for a tv we should not waste time looking in the bathroom. It is your job to point this out. 

You should always provide reasoning along with a number identifying where we should not explore. If there are multiple right answers you should separate them with commas. Always include Reasoning: <your reasoning> and Answer: <your answer(s)>. If there are no suitable answers leave the space after Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I avoid spending time searching if I am looking for a knife?

Assistant:
Reasoning: Cluster 1 contains items that are likely part of an entertainment room. Cluster 2 contains objects that are likely part of an office room and cluster 3 contains items likely found in a kitchen. A knife is not likely to be in an entertainment room or an office room so we should avoid searching those spaces.
Answer: 1,2


Other considerations 

1. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgment when determining what room a cluster of objects is likely to be in.
2. Provide reasoning for each cluster before giving the final answer
"""

V2_SYSTEM_PROMPT_POSITIVE = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should explore next. For example if we are in a house and looking for a tv we should explore areas that typically have tv's such as bedrooms and living rooms.

You should always provide reasoning along with a number identifying where we should explore. If there are multiple right answers you should separate them with commas. Always include Reasoning: <your reasoning> and Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I search next if I am looking for a knife?

Assistant:
Reasoning: Cluster 1 contains items that are likely part of an entertainment room. Cluster 2 contains objects that are likely part of an office room and cluster 3 contains items likely found in a kitchen. Because we are looking for a knife which is typically located in a kitchen, so we should check cluster 3.
Answer: 3


Other considerations 

1. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgment when determining what room a cluster of objects is likely to be in.
2. Provide reasoning for each cluster before giving the final answer.
3. Feel free to think multiple steps in advance; for example if one room is typically located near another then it is ok to use that information to provide higher scores in that direction.
"""

V2_SYSTEM_PROMPT_NEGATIVE_NO_REASONING = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should not waste time exploring. For example if we are in a house and looking for a tv we should not waste time looking in the bathroom. It is your job to point this out. 

You should always provide a number identifying where we should not explore. If there are multiple right answers you should separate them with commas. Always include Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I avoid spending time searching if I am looking for a knife?

Assistant:
Answer: 1,2


Other considerations 

1. Disregard the frequency of the objects listed on each line. If there are multiple of the same item in  a cluster it will only be listed once in that cluster.
2. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgment when determining what room a cluster of objects is likely to be in.
"""

V2_SYSTEM_PROMPT_POSITIVE_NO_REASONING = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should explore next. For example if we are in a house and looking for a tv we should explore areas that typically have tv's such as bedrooms and living rooms.

You should always provide a number identifying where we should explore. If there are multiple right answers you should separate them with commas. Always include Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I search next if I am looking for a knife?

Answer: 3


Other considerations 

1. Disregard the frequency of the objects listed on each line. If there are multiple of the same item in  a cluster it will only be listed once in that cluster.
2. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgment when determining what room a cluster of objects is likely to be in.
"""

def generate_options(object_clusters, indeces=None, explored=None, objects=True):
    options = ""
    if indeces is None:
        indeces = range(len(object_clusters))
    for i in indeces:
        options += f"- Cluster {i + 1}"
        if explored is not None:
            options += f" ({'explored' if explored[i] else 'not explored'})"
        if objects:
            objects_string = ""
            for ob in object_clusters[i]:
                objects_string += ob + ", "
            options += f": {objects_string[:-2]}"
        options += "\n"
    return options


class LLMClusterScorer:

    def __init__(self, client: OpenAI, goal, model="gpt-4o"):

        self.client = client
        self.goal = goal
        self.model = model


    def score_clusters(self, object_clusters):

        # Convert object clusters to a tuple of tuples so we can hash it and get unique elements
        object_clusters_tuple = [tuple(x) for x in object_clusters]
        # Remove empty clusters and duplicate clusters
        object_clusters = list(set(tuple(object_clusters_tuple)) - set({tuple([])}))

        answer_counts, reasonings = self.choose_clusters(object_clusters, positives=True)
        positive_language_scores = np.zeros(len(object_clusters_tuple))
        for key, value in answer_counts.items():
            for i, x in enumerate(object_clusters_tuple):
                if x == object_clusters[key - 1]:
                    positive_language_scores[i] = value

        answer_counts, reasonings = self.choose_clusters(object_clusters, positives=False)
        negative_language_scores = np.zeros(len(object_clusters_tuple))
        for key, value in answer_counts.items():
            for i, x in enumerate(object_clusters_tuple):
                if x == object_clusters[key - 1]:
                    negative_language_scores[i] = value
        
        return positive_language_scores - negative_language_scores
    

    @retry.retry(tries=5)
    def choose_clusters(self, object_clusters, env="a house", positives=True, num_samples=5, reasoning_enabled=True):

        if reasoning_enabled:
            if positives:
                system_message = V2_SYSTEM_PROMPT_POSITIVE
            else:
                system_message = V2_SYSTEM_PROMPT_NEGATIVE
        else:
            if positives:
                system_message = V2_SYSTEM_PROMPT_POSITIVE_NO_REASONING
            else:
                system_message = V2_SYSTEM_PROMPT_NEGATIVE_NO_REASONING


        messages = [
            {"role": "system", "content": system_message}
        ]
        if len(object_clusters) > 0:
            options = generate_options(object_clusters)
            if positives:
                messages.append({"role": "user", "content": f"I observe the following clusters of objects while exploring {env}:\n\n{options}\nWhere should I search next if I am looking for {self.goal}?"})
            else:
                messages.append({"role": "user", "content": f"I observe the following clusters of objects while exploring {env}:\n\n{options}\nWhere should I avoid spending time searching if I am looking for {self.goal}?"})
            completion = self.client.chat.completions.create(
                model=self.model, temperature=1,
                n=num_samples, messages=messages)
            answers = []
            reasonings = []
            for choice in completion.choices:
                try:
                    complete_response = choice.message.content
                    # Make the response all lowercase
                    complete_response = complete_response.lower()
                    if reasoning_enabled:
                        reasoning = complete_response.split("reasoning: ")[1].split("\n")[0]
                    else:
                        reasoning = "disabled"
                    # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
                    if len(complete_response.split("answer:")) > 1:
                        answer = complete_response.split("answer:")[1].split("\n")[0]
                        # Separate the answers by commas
                        answers.append([int(x) for x in answer.split(",")])
                    else:
                        answers.append([])
                    reasonings.append(reasoning)
                except:
                    answers.append([])
            # Flatten answers
            flattened_answers = [item for sublist in answers for item in sublist]
            # It is possible GPT gives an invalid answer less than 1 or greater than 1 plus the number of object clusters. Remove invalid answers
            filtered_flattened_answers = [x for x in flattened_answers if x >= 1 and x <= len(object_clusters)]
            # Aggregate into counts and normalize to probabilities
            answer_counts = {x: filtered_flattened_answers.count(x) / len(answers) for x in set(filtered_flattened_answers)}

            return answer_counts, reasonings
        raise Exception("Object categories must be non-empty")