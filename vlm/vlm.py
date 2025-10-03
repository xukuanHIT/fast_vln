import os
from glob import glob
import time

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np

from typing import Tuple, Optional, Union
from collections import defaultdict

from planner.tsdf_planner import TSDFPlanner, Frontier
from map.map import Map


class VLM:
    def __init__(self, cfg,):

        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.log_dir = None
        self.step = 0

    def update_save_dir(self, log_dir):
        self.log_dir = log_dir

    def update_step(self, step):
        self.step = step


    def chat(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text
    

    def format_question(self, step):
        question = step["question"]
        image_goal = None
        if "task_type" in step and step["task_type"] == "image" and os.path.exists(step["image"]):
            image_goal = Image.open(image_goal)

        return question, image_goal


    # def format_prefiltering_prompt(self, question, class_list, top_k=10, image_goal=None):
    #     system_prompt = "You are an AI agent in a 3D indoor scene.\n"
    #     prompt = "You need to find, from a list of object categories, the categories that are most relevant to a given question or description. These selected objects should be able to, or are likely to, answer the question or best match the description.\n"
            
    #     prompt += "You need provide your answer in four lines.\n"
    #     prompt += "Line 1: The object categories from the list that should be able to directly and immediately answer the question or complete the described task. (The target object you are looking for)\n"
    #     prompt += "Line 2: The reasons for selecting the objects in line 1.\n"
    #     prompt += "Line 3: The object categories from the list that are indirectly relevant to the question/description. (Seeing these objects might suggest that the directly relevant objects are nearby.)\n"
    #     prompt += "Line 4: The reasons for selecting the objects in line 3.\n"

    #     prompt += "These are the rules for the task.\n"
    #     prompt += "1. The first and third lines should be the name of the selected object categories, separated by comma. Do not include any additional information in these two lines.\n"
    #     prompt += "2. Read through the whole object list. For the objects in the first line. If none are suitable, you may include objects outside the list.\n"
    #     prompt += "3. For the objects in the third line, ONLY output categories that appear in the provided list. The output names must exactly match the list (no synonyms or invented words). If the object is not in the list but a broader or closest category is, select that category.\n"
    #     prompt += "4. Rank objects in the first and third lines based on how well they can help your exploration given the question or description.\n"
    #     prompt += "5. If multiple categories are relevant, output all of them.\n"
    #     prompt += "6. The objects in the first and third lines must not overlap. If you see an object that is related to the question or description, but seeing this object does not directly indicate the target object—only that it is related or that the target object might be nearby—then put it in the third line, and do not put it in the first line.\n"


    #     prompt += "Following is the concrete content of the task and you should retrieve helpful objects in order:\n"
    #     prompt += f"Question: {question}"
    #     prompt += "Following is a list of objects that you can choose, each object one line\n"


    #     for i, cls in enumerate(class_list):
    #         prompt += f"{cls}\n"

    #     return system_prompt, prompt


    # def format_prefiltering_prompt(self, question, class_list, top_k=10, image_goal=None):
    #     system_prompt = "You are an AI agent in a 3D indoor scene.\n"
    #     prompt = "You need to parse a task/question description to identify an object that must be found in order to answer the question or complete the task."
    #     prompt = "You also need to find, from a list of object categories, the categories that are most relevant to a given question or description. These selected objects should be able to, or are likely to, answer the question or best match the description.\n"
            
    #     prompt += "You need provide your answer in four lines.\n"
    #     prompt += "Line 1: The object you parsed from the task/question description. Do not provide any other information except the object name.\n"
    #     prompt += "Line 2: The reasons for selecting the object in line 1.\n"
    #     prompt += "Line 3: The object categories from the list that are relevant to the question/description. (Seeing these objects might suggest that the directly relevant objects are nearby.)\n"
    #     prompt += "Line 4: The reasons for selecting the objects in line 3.\n"

    #     prompt += "These are the rules for the task.\n"
    #     prompt += "1. The third lines should be the name of the selected object categories, separated by comma. Do not include any additional information in the third lines.\n"
    #     prompt += "2. For the objects in the third line, ONLY output categories that appear in the provided list. The output names must exactly match the list (no synonyms or invented words). If the object is not in the list but a broader or closest category is, select that category.\n"
    #     prompt += "3. Rank objects in the third lines based on how well they can help your exploration given the question or description.\n"
    #     prompt += "4. If multiple categories are relevant, output all of them.\n"


    #     prompt += "Following is the concrete content of the task/question description:\n"
    #     prompt += f"Question: {question}"
    #     prompt += "Following is a list of objects that you can choose for selecting objects in the third line, each object one line\n"


    #     for i, cls in enumerate(class_list):
    #         prompt += f"{cls}\n"

    #     return system_prompt, prompt


    def format_prefiltering_prompt(self, question, class_list, top_k=10, image_goal=None):
        system_prompt = "You are an AI agent in a 3D indoor scene.\n"
        prompt = "You need to parse a task/question description, and select an object from a list of object categories that must be found in order to answer the question or complete the task."
        prompt = "You also need to find, from a list of object categories, the categories that are most relevant to a given question or description. These selected objects should be able to, or are likely to, answer the question or best match the description.\n"
            
        prompt += "You need provide your answer in four lines.\n"
        prompt += "Line 1: The object you select from the provided list to answer the question or complete the task. Do not provide any other information except the object name.\n"
        prompt += "Line 2: The reasons for selecting the object in line 1.\n"
        prompt += "Line 3: The object categories from the list that are relevant to the question/description. (Seeing these objects might suggest that the directly relevant objects are nearby.)\n"
        prompt += "Line 4: The reasons for selecting the objects in line 3.\n"

        prompt += "These are the rules for the task.\n"
        prompt += "1. The third lines should be the name of the selected object categories, separated by comma. Do not include any additional information in the third lines.\n"
        prompt += "2. For the objects in the first and third line, ONLY output categories that appear in the provided list. The output names must exactly match the list (no synonyms or invented words). If the object is not in the list but a broader or closest category is, select that category.\n"
        prompt += "3. Rank objects in the third lines based on how well they can help your exploration given the question or description.\n"
        prompt += "4. If multiple categories are relevant, output all of them.\n"


        prompt += "Following is the concrete content of the task/question description:\n"
        prompt += f"Question: {question}"
        prompt += "Following is a list of objects that you can choose for selecting objects in the first and third lines, each object one line\n"


        for i, cls in enumerate(class_list):
            prompt += f"{cls}\n"

        return system_prompt, prompt




    def parse_target_objects(self, question, class_list):
        parsing_sys_prompt, parsing_content = self.format_prefiltering_prompt(question, class_list)
        prefiltering_messages = [
            {"role": "system", "content": parsing_sys_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": parsing_content}],
            }
        ]

        response = self.chat(prefiltering_messages)
        if response is None:
            return None

        response = response[0]
        lines = response.strip().split("\n")
        if len(lines) != 4:
            return None
        
        target_objects = lines[0].strip().split(",")
        target_objects = [target_object.strip() for target_object in target_objects]

        relevant_objects = lines[2].strip().split(",")
        relevant_objects = [relevant_object.strip() for relevant_object in relevant_objects]

        if len(target_objects) + len(relevant_objects) < 1:
            return None

        target_reason, relevant_reason = lines[1].strip(), lines[3].strip()

        provide_class_set, relevant_class_set, target_class_set = set(class_list), set(relevant_objects), set(target_objects)

        relevant_class_set = provide_class_set & relevant_class_set
        target_class_set = target_class_set - relevant_class_set
        new_target_class_set = target_class_set - provide_class_set


        if len(target_class_set) < 1 and len(target_objects) > 0:
            target_class_set = set([target_objects[0]])

        return [new_target_class_set, target_class_set, target_reason, relevant_class_set, relevant_reason]



    def format_task_checking_prompt(self, question, images, objects):
        system_prompt = "You are an AI agent in a 3D indoor scene.\n"

        content = []

        text = "You need to explore a scene in order to answer a given question or accomplish a given task.\n"
        text += "You must decide whether the provided images and objects are sufficient to answer the task. You must then select one image and one object most relevant to the task.\n"

        text += "Following is the concrete content of the task and you should retrieve helpful objects in order:\n"
        text += f"Question/Task: {question}"

        text += "Following are all the images that you can choose:\n"
        content.append({"type": "text", "text": text})

        for i in range(len(images)):
            text = f"The following is Image {i}:\n"
            content.append({"type": "text", "text": text})
            content.append({"type": "image","image": Image.fromarray(images[i])})
            
            text = f"The following are the objects contained in Image {i}:\n"
            for object_label in objects[i]:
                text += f"{object_label},"
            text += "\n"
            content.append({"type": "text", "text": text})

        text = "You need provide your answer in three lines.\n"
        text += "Line 1: Choose one option only: 1 = From the provided images/objects, I can answer the question or complete the task, 2 = From the provided images/objects, I cannot answer the question or complete the task."
        text += "Do not output anything else beyond 1 or 2 in the Line 1.\n"
        text += "Line 2: The index of the most relevant image and the name of the most relevant object from that image."
        text += "Format of Line 2: image_index, object_name. Only use objects explicitly provided for that image. Do not invent or mention any object that is not in the given list.\n"
        text += "Line 3: A short explanation of why this image and object were chosen."
        content.append({"type": "text", "text": text})


        text = "These are the rules for the task.\n"
        text += "1. The format of your response must strictly follow the above requirements. The first line should contain only one number, either 0 or 1, and nothing else.\n"
        text += "2. The second line should contain only the index of the most relevant image and the name of the most relevant object from that image, separated by a comma. Only use objects explicitly provided for that image. Do not invent or mention any object that is not in the given list.\n"
        text += "3. If the task is to answer a question, you need to provide the answer to the question in the third line.\n"
        content.append({"type": "text", "text": text})


        text = "Here is an example of the required answer format:\n"
        text += "1\n"
        text += "2, teddy bear\n"
        text += "The teddy bear in image 2 directly answers the question 'Where is the teddy bear?'\n"
        content.append({"type": "text", "text": text})

        return system_prompt, content


    def check_task_finished(self, question, images, objects):
        checking_sys_prompt, checking_content = self.format_task_checking_prompt(question, images, objects)
        prefiltering_messages = [
            {"role": "system", "content": checking_sys_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": checking_content}],
            }
        ]


        response = self.chat(prefiltering_messages)

        print("check_task_finished: {}".format(response))


        if response is None:
            return None

        response = response[0]
        lines = response.strip().split("\n")
        if len(lines) < 1:
            return None
    
        if int(lines[0]) != 1:
            return None
        
        if len(lines) != 3:
            return None
        
        image_idx, target_object = lines[1].split(",")[0].strip(), lines[1].split(",")[1].strip()
        image_idx = int(image_idx)

        if image_idx < 0 or image_idx > len(images) -1:
            return None

        return [image_idx, target_object, lines[2]]


    def query_vlm_for_response(
        self,
        question: str,
        scene: Map,
        tsdf_planner: TSDFPlanner,
        cfg,
        verbose: bool = False,
    ):
        
        target_objects, relevant_objects, target_keyframes, relevant_keyframes = scene.find_target()

        print("len(target_objects) ================= {}".format(len(target_objects)))

        if len(target_objects) > 0:
            
            class_to_kf_ids_set = defaultdict(set)
            for target_object in target_objects:
                object_class = target_object.get_class_label()
                class_to_kf_ids_set[object_class].update(target_object.observers) 

            frame_to_objects = defaultdict(set)
            for class_label, kf_ids in class_to_kf_ids_set.items():
                for kf_id in kf_ids:
                    frame_to_objects[kf_id].add(class_label)

            keyframe_ids, objects_in_keyframes = list(frame_to_objects.keys()), list(frame_to_objects.values())
            images = [scene.keyframes[kf_id].image for kf_id in keyframe_ids]

            response = self.check_task_finished(question, images, objects_in_keyframes)


            if response is None:
                scene.target_manager.add_keyframes_to_blacklist(set(keyframe_ids))
                target_object_ids = [target_object.object_id for target_object in target_objects]
                scene.target_manager.add_objects_to_blacklist(set(target_object_ids))
            else:
                image_idx, target_class, reason = response

                final_keyframe_id = keyframe_ids[image_idx]
                final_objects = []
                for target_object in target_objects:
                    if target_object.get_class_label() == target_class and final_keyframe_id in target_object.observers:
                        final_objects.append(target_object)

                final_confidences = [final_object.get_confidence() for final_object in final_objects]
                max_idx = np.argmax(np.array(final_confidences))
                return True, final_objects[max_idx], scene.keyframes[final_keyframe_id], reason
            

        # didn't find target class, search for frontiers
        target_frontiers, relevant_frontiers = [], []
        for frontier in tsdf_planner.frontiers:
            class_labels_in_frontier = frontier.frame.detections.class_label_set
            target_classes_in_frontier = class_labels_in_frontier & scene.target_manager.target_class_set 
            if len(target_classes_in_frontier) > 0:
                target_frontiers.append(frontier)
            else:
                relevant_classes_in_frontier = class_labels_in_frontier & scene.target_manager.relevant_class_set
                if len(relevant_classes_in_frontier) > 0:
                    relevant_frontiers.append(frontier)


        frontier_pool, reason = [], ""
        if len(target_frontiers) > 0:
            frontier_pool = target_frontiers
            reason = "Select the frontier which contains the target objects: {}".format(list(scene.target_manager.target_class_set))
        elif len(relevant_frontiers) > 0:
            frontier_pool = relevant_frontiers
            reason = "Select the frontier which contains the relevant objects: {}".format(list(scene.target_manager.relevant_class_set))
        else:
            frontier_pool = tsdf_planner.frontiers
            reason = "Select the frontier using clip"

        frontier_fts = []
        for frontier in frontier_pool:
            if frontier.frame.clip_ft is None:
                frontier_ft = scene.encode_image_with_clip([frontier.frame.image])
                frontier.frame.clip_ft = frontier_ft[0]
            
            frontier_fts.append(frontier.frame.clip_ft)

        frontier_fts = np.stack(frontier_fts)  
        similarities = frontier_fts @ scene.target_manager.task_clip_ft
        best_idx = np.argmax(similarities)

        return False, frontier_pool[best_idx], frontier_pool[best_idx].frame.image, reason


