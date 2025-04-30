from openai import OpenAI
import retry
import re
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

ROOM_DETECTOR_SYSTEM_PROMPT = """You are a robot exploring an environment for the first time. Based on a series of observations, you have to guess the rooms that exist in the environment. Observations will be given as a list of object clusters numbered 1 to N. You might also be given a list of the rooms that was generated previously, in which case, you will improve that list based on the current objects. For each room, you must mention the objects that it contains. You don't have to assign an object to any room if it is not informative enough.

Strict Answer Format:

- Reasoning: <your detailed reasoning>
- Answer: <list of rooms and their objects>

Example

User Input:
I am observing the following clusters of objects:

- Cluster 1: air fryer, oven
- Cluster 2: bed, closet, bookshelf
- Cluster 3: toilet, bathtub, wash basin
- Cluster 4: boxes, old newspapers

This is the list of the rooms that I have previously come up with:

- Entertainment room: sofa, tv, speaker
- Office room: desk, chair, computer
- Kitchen: sink, microwave, refrigerator

Use your current observations to decide if any rooms have to be added to the list or if the list has to be revised. If an object does not clearly belong to a known type of room, do not assign it. Provide reasoning before generating the list.

Assistant (Model):
Reasoning:

- Air fryer and oven likely belong to the Kitchen.
- Bed, closet, and bookshelf strongly suggest a Bedroom.
- Toilet, bathtub, and wash basin suggest a Washroom.
- Boxes and old newspapers do not clearly indicate any specific type of room and could be generic storage. Without more information, it is best not to assign them to any particular room.

Answer:

- Entertainment room: sofa, tv, speaker
- Office room: desk, chair, computer
- Kitchen: sink, microwave, refrigerator, air fryer, oven
- Bedroom: bed, closet, bookshelf
- Washroom: toilet, bathtub, wash basin
- Unassigned: boxes, old newspapers
"""

MAIN_SYSTEM_PROMPT = """You are a robot exploring an environment. You have a list of rooms and their objects, and you will be given a list of clusters of objects. Your task is to decide which cluster you should explore next to find a given object.

Considerations:

1. Feel free to think multiple steps ahead. For example, if it is known that certain rooms are often located near the type of room where the object might be found, you can use that information to make a more informed guess.
2. Always provide reasoning for your choice, followed by your final answer.

Answer Format:

- Reasoning: <your detailed reasoning>
- Answer: <cluster number>

Example

User:
This is the list of the rooms:

- Entertainment room: sofa, tv, speaker
- Office room: desk, chair, computer
- Kitchen: sink, microwave, refrigerator, air fryer, oven
- Bedroom: bed, closet, bookshelf
- Washroom: toilet, bathtub, wash basin
- Unassigned: boxes, old newspapers

I am showing you the following clusters to choose from:

- Cluster 1: air fryer, oven
- Cluster 2: bed, closet, bookshelf
- Cluster 3: toilet, bathtub, wash basin
- Cluster 4: boxes, old newspapers

I am looking for a knife. Where should I explore next?

Assistant (Model):
Reasoning:

- The items in Cluster 1 are associated with a kitchen. Knives are commonly found in kitchens, so this is a strong match.
- The items in Cluster 2 suggest a bedroom, and items in Cluster 3 appear to be in a washroom, which are less likely to have a knife.
- The items in Cluster 4 are unassigned and do not strongly suggest a kitchen setting.

Since Cluster 1 is most likely to contain a knife, choose Cluster 1.

Answer: 1
"""


def find_first_integer(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    else:
        raise ValueError('No integer found in string')


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


class LLMRoomDetector:

    def __init__(self, client: OpenAI, model="gpt-4o"):

        self.client = client
        self.model = model
        self.previous_rooms_list = ""
    

    @retry.retry(tries=5)
    def detect_rooms(self, object_clusters):

        messages = [
            {"role": "system", "content": ROOM_DETECTOR_SYSTEM_PROMPT}
        ]
        options = generate_options(object_clusters)
        if self.previous_rooms_list:    
            messages.append({"role": "user", "content": f"I am observing the following clusters of objects:\n\n{options}\nThis is the list of the rooms that I have previously come up with:\n\n{self.previous_rooms_list}\n\nUse your current observations to decide if any rooms have to be added to the list or if the list has to be revised. If an object does not clearly belong to a known type of room, do not assign it. Provide reasoning before generating the list."})
        else:
            messages.append({"role": "user", "content": f"I am observing the following clusters of objects:\n\n{options}\nGenerate the most likely list of the rooms that these objects reside in. If an object does not clearly belong to a known type of room, do not assign it. Provide reasoning before generating the list."})
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages)
        complete_response = completion.choices[0].message.content
        answer = complete_response.split("Answer:")[1].strip()
        self.previous_rooms_list = answer
        return answer


class LLMAgentWithRoomDetector:

    def __init__(self, client: OpenAI, goal, model="gpt-4o"):

        self.client = client
        self.goal = goal
        self.model = model
        self.room_detector = LLMRoomDetector(client, model)
    
    
    @retry.retry(tries=5)
    def choose_cluster(self, object_clusters):

        messages = [
            {"role": "system", "content": MAIN_SYSTEM_PROMPT}
        ]
        if len(object_clusters) > 0:
            rooms_list = self.room_detector.detect_rooms(object_clusters)
            options = generate_options(object_clusters)
            messages.append({"role": "user", "content": f"This is the list of the rooms:\n\n{rooms_list}\n\nI am showing you the following clusters to choose from:\n\n{options}\nI am looking for {self.goal}. Where should I explore next?"})
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages)
            complete_response = completion.choices[0].message.content
            # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
            answer = int(find_first_integer(complete_response.lower().split("answer")[1])) - 1
            return answer
        raise Exception("There must be at least one object cluster")
