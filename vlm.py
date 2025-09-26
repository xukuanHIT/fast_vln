import os
from glob import glob
import time

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

import logging
from typing import Tuple, Optional, Union

from tsdf_planner import TSDFPlanner, SnapShot, Frontier
from map import Map


class VLM:
    def __init__(self, cfg,):

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

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


    def format_prefiltering_prompt(self, question, class_list, top_k=10, image_goal=None):
        content = []
        sys_prompt = "You are an AI agent in a 3D indoor scene.\n"
        prompt = "Your goal is to answer questions about the scene through exploration.\n"
        prompt += "To efficiently solve the problem, you should first rank objects in the scene based on their importance.\n"
        prompt += "These are the rules for the task.\n"
        prompt += "1. Read through the whole object list.\n"
        prompt += "2. Rank objects in the list based on how well they can help your exploration given the question.\n"
        prompt += f"3. Reprint the name of all objects that may help your exploration given the question. "
        prompt += "4. Do not print any object not included in the list or include any additional information in your response.\n"
        # content.append((prompt,))
        # ------------------format an example-------------------------
        # prompt = "Here is an example of selecting helpful objects:\n"
        prompt += "Here is an example of selecting helpful objects:\n"
        prompt += "Question: What can I use to watch my favorite shows and movies?\n"
        prompt += (
            "Following is a list of objects that you can choose, each object one line\n"
        )
        prompt += "painting\nspeaker\nbox\ncabinet\nlamp\ntv\nbook rack\nsofa\noven\nbed\ncurtain\n"
        prompt += "Answer: tv\nspeaker\nsofa\nbed\n"
        # content.append((prompt,))
        # ------------------Task to solve----------------------------
        # prompt = f"Following is the concrete content of the task and you should retrieve helpful objects in order:\n"
        prompt += f"Following is the concrete content of the task and you should retrieve helpful objects in order:\n"
        prompt += f"Question: {question}"
        # if image_goal is not None:
        #     content.append((prompt, image_goal))
        #     content.append(("\n",))
        # else:
        #     content.append((prompt + "\n",))
        # prompt = (
        #     "Following is a list of objects that you can choose, each object one line\n"
        # )
        prompt += f"Following is a list of objects that you can choose, each object one line\n"
        for i, cls in enumerate(class_list):
            prompt += f"{cls}\n"
        prompt += "Answer: "
        # content.append((prompt,))
        return sys_prompt, prompt


    def prefiltering(
        self, question, seen_classes, top_k=10, image_goal=None, verbose=False
    ):
        '''
        prefiltering, 返回snapshot_classes
        '''
        # 构建 prompt, 返回值分别是系统prompt, 和当前问题的prompt
        prefiltering_sys_prompt, prefiltering_content = self.format_prefiltering_prompt(
            question, sorted(list(seen_classes)), top_k=top_k, image_goal=image_goal
        )

        prefiltering_messages = [
            {"role": "system", "content": prefiltering_sys_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": prefiltering_content}],
            }
        ]

        response = self.chat(prefiltering_messages)
        if response is None:
            return []

        # 后处理OpenAI回复, 抽取相关物体
        # parse the response and return the top_k objects
        response = response[0]
        selected_classes = response.strip().split("\n")
        selected_classes = [cls.strip() for cls in selected_classes]
        selected_classes = [cls for cls in selected_classes if cls in seen_classes]
        selected_classes = selected_classes[:top_k]

        if verbose:
            logging.info(f"Prefiltering selected classes: {selected_classes}")


        return selected_classes



    def get_step_info(self, step, verbose=False):
        '''
        整理query所需的信息, 编码数据, prefoltering. 返回值:
        '''

        # 获取问题数据, question是文本问题, image_goal是base64格式的图像数据
        # 只有task_type是image时, 才会有image_goal
        # 1 get question data
        question, image_goal = self.format_question(step)

        # 
        # 1 prefiltering, note that we need the obj_id_mapping
        if step.get("use_prefiltering") is True:
            class_labels = set(step["keyframe_objects"].keys()) | set(step["frontier_objects"].keys())
            if len(class_labels) > step["top_k_categories"] and (len(step["keyframe_images"]) + len(step["frontier_images"])) > 10:
                # 论文中的prefiltering, 这时的snapshot_classes里只包含ChatGPT推测的和question相关的object
                keep_classes = self.prefiltering(
                    question=question,
                    seen_classes=class_labels,
                    top_k=step["top_k_categories"],
                    image_goal=image_goal,
                    verbose=verbose,
                )
                print("question == {}".format(question))
                print("class_labels == {}".format(class_labels))
                print("keep_classes == {}".format(keep_classes))

                keep_keyframe_objects, keep_frontier_objects = {}, {}
                for keep_class in keep_classes:
                    if keep_class in step["keyframe_objects"]:
                            keep_keyframe_objects[keep_class] = step["keyframe_objects"][keep_class]

                    # if keep_class in step["frontier_objects"]:
                    #     keep_frontier_objects[keep_class] = step["frontier_objects"][keep_class]

                # step["keyframe_objects"], step["frontier_objects"] = keep_keyframe_objects, keep_frontier_objects
                step["keyframe_objects"] = keep_keyframe_objects

                if verbose:
                    logging.info(
                        f"Prefiltering class labels: {len(class_labels)} -> {len(keep_classes)}"
                    )


        # 把 frontier 图片转成 base64 格式
        # frontier_imgs 是 frontier 对应的 resize后的图片
        # 2.2 get frontiers
        frontier_imgs, frontier_ids, frontier_added_ids  = [], [], set()
        for _, frontier_id_set in step["frontier_objects"].items():
            for frontier_id in frontier_id_set:
                if frontier_id not in frontier_added_ids:
                    frontier_imgs.append(step["frontier_images"][frontier_id])
                    frontier_ids.append(frontier_id)
                    frontier_added_ids.add(frontier_id)


        keyframe_imgs, keyframe_ids, keyframe_added_ids  = [], [], set()
        for class_label, keyframe_id_set in step["keyframe_objects"].items():
            for keyframe_id in keyframe_id_set:
                if keyframe_id not in keyframe_added_ids:
                    keyframe_imgs.append(step["keyframe_images"][keyframe_id])
                    keyframe_ids.append(keyframe_id)
                    keyframe_added_ids.add(keyframe_id)

        print("keep keyframe = {}, keep frontier = {}".format(len(keyframe_added_ids), len(frontier_added_ids)))

        return question, image_goal, frontier_imgs, frontier_ids, keyframe_imgs, keyframe_ids




    def format_explore_prompt(
        self,
        question,
        frontier_imgs,
        keyframe_imgs,
        image_goal=None,
    ):
        sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment. To answer the question, you are required to choose either a Snapshot as the answer or a Frontier to further explore.\n"
        sys_prompt += "Definitions:\n"
        sys_prompt += "Snapshot: A focused observation of several objects. Choosing a Snapshot means that this snapshot image contains enough information for you to answer the question. "
        sys_prompt += "If you choose a Snapshot, you need to directly give an answer to the question. If you don't have enough information to give an answer, then don't choose a Snapshot.\n"
        sys_prompt += "Frontier: An observation of an unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction. "
        sys_prompt += "If you choose a Frontier, you need to explain why you would like to choose that direction to explore.\n"

        content = []
        # 1 first is the question
        text = f"Question: {question}\n"
        content.append({"type": "text", "text": text})
        if image_goal is not None:
            content.append({"type": "image", "image": image_goal})

        text = "Select the Frontier/Snapshot that would help find the answer of the question.\n"
        content.append({"type": "text", "text": text})

        # 3 here is the snapshot images
        # text = "The followings are all the snapshots that you can choose (followed with contained object classes)\n"
        # text += "Please note that the contained classes may not be accurate (wrong classes/missing classes) due to the limitation of the object detection model. "
        # text += "So you still need to utilize the images to make decisions.\n"
        text = "The followings are all the snapshots that you can choose\n"
        content.append({"type": "text", "text": text})
        if len(keyframe_imgs) == 0:
            content.append({"type": "text", "text": "No Snapshot is available\n"})
        else:
            for i in range(len(keyframe_imgs)):
                content.append({"type": "text", "text": "The following is Snapshot {}".format(i)})
                content.append({"type": "image","image": Image.fromarray(keyframe_imgs[i])})

                # if use_snapshot_class:
                #     text = ", ".join(snapshot_classes[i])
                #     content.append((text,))
                # content.append(("\n",))

        # 4 here is the frontier images
        text = "The followings are all the Frontiers that you can explore: \n"
        content.append({"type": "text", "text": text})
        if len(frontier_imgs) == 0:
            content.append({"type": "text", "text": "No Frontier is available\n"})
        else:
            for i in range(len(frontier_imgs)):
                content.append({"type": "text", "text": "The following is Frontier {}".format(i)})
                content.append({"type": "image","image": Image.fromarray(frontier_imgs[i])})

        print("keyframe_num = {}, frontier_num = {}".format(len(keyframe_imgs), len(frontier_imgs)))


        # 5 here is the format of the answer
        text = "Please provide your answer in the following format: 'Snapshot i\n[Answer]' or 'Frontier i\n[Reason]', where i is the index of the snapshot or frontier you choose. "
        text += "For example, if you choose the first snapshot, you can return 'Snapshot 0\nThe fruit bowl is on the kitchen counter.'. "
        text += "If you choose the second frontier, you can return 'Frontier 1\nI see a door that may lead to the living room.'.\n"
        text += "Note that if you choose a snapshot to answer the question, (1) you should give a direct answer that can be understood by others. Don't mention words like 'snapshot', 'on the left of the image', etc; "
        text += "(2) you can also utilize other snapshots, frontiers and egocentric views to gather more information, but you should always choose one most relevant snapshot to answer the question.\n"
        content.append({"type": "text", "text": text})

        return sys_prompt, content



    def explore_step(self, step, cfg, verbose=False):
        step["use_prefiltering"] = cfg.prefiltering
        step["top_k_categories"] = cfg.top_k_categories

        question, image_goal, frontier_imgs, frontier_ids, keyframe_imgs, keyframe_ids = self.get_step_info(step, verbose)      

        print("select frontiers === {}, select keyframes === {}".format(frontier_ids, keyframe_ids))


        # 整理prompt
        sys_prompt, content = self.format_explore_prompt(question, frontier_imgs, keyframe_imgs, image_goal=image_goal)

        if verbose:
            logging.info(f"Input prompt:")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)


        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content,},
        ]

        full_response = self.chat(messages)
        full_response = full_response[0]


        final_response, final_reason, response_valid = None, None, True
        if full_response is None:
            print("VLM returns None.")
        else:
            print("full_response ================= {}".format(full_response))
            full_response = full_response.strip()
            if "\n" in full_response:
                full_response = full_response.split("\n")
                response, reason = full_response[0], full_response[-1]
                response, reason = response.strip(), reason.strip()
            else:
                response = full_response
                reason = ""
            response = response.lower()
            try:
                choice_type, choice_id = response.split(" ")
            except Exception as e:
                print(f"Error in splitting response: {response}")
                print(e)
                response_valid = False

            if response_valid:
                print("choice_type = {}, choice_id = {}, max_id = {}".format(choice_type, choice_id, len(frontier_imgs)))
                if choice_type == "snapshot" and choice_id.isdigit() and 0 <= int(choice_id) < len(keyframe_imgs):
                    response_valid = True
                elif choice_type == "frontier" and choice_id.isdigit() and 0 <= int(choice_id) < len(frontier_imgs):
                    response_valid = True
                else:
                    response_valid = False

            if response_valid:
                final_response = response
                final_reason = reason

        return final_response, frontier_ids, keyframe_ids, final_reason




    def query_vlm_for_response(
        self,
        question: str,
        scene: Map,
        tsdf_planner: TSDFPlanner,
        cfg,
        verbose: bool = False,
    ) -> Optional[Tuple[Union[SnapShot, Frontier], str, int]]:
        # prepare input for vlm
        step_dict = {}

        # prepare question
        step_dict["question"] = question

        # prepare keyframes
        keyframe_objects, keyframe_images = {}, {}
        for _, obj in scene.objects_3d.items():
            if obj.class_name not in keyframe_objects:
                keyframe_objects[obj.class_name] = set()

            keyframe_objects[obj.class_name].update(obj.observers)

        for kf_id, kf in scene.keyframes.items():
            keyframe_images[kf_id] = kf.image

        # prepare frontiers
        frontier_objects, frontier_images, frontier_id_to_index = {}, {}, {}
        for i, frontier in enumerate(tsdf_planner.frontiers):
            object_ids = frontier.frame.class_ids
            class_labels = [scene.obj_classes.get_classes_arr()[obj_id] for obj_id in object_ids]
            for class_label in class_labels:
                if class_label in frontier_objects:
                    frontier_objects[class_label].add(frontier.frontier_id)
                else:
                    frontier_objects[class_label] = set([frontier.frontier_id])

            frontier_images[frontier.frontier_id] = frontier.frame.image
            frontier_id_to_index[frontier.frontier_id] = i





        # debug visualization
        for kf_id, img in keyframe_images.items():
            # 转 PIL 再保存
            image_name = "step_{}_kf_id_{}.png".format(self.step, kf_id)
            im_pil = Image.fromarray(img)
            save_path = os.path.join(self.log_dir, image_name)
            im_pil.save(save_path)
        for ft_id, img in frontier_images.items():
            # 转 PIL 再保存
            image_name = "step_{}_ft_id_{}.png".format(self.step, ft_id)
            im_pil = Image.fromarray(img)
            save_path = os.path.join(self.log_dir, image_name)
            im_pil.save(save_path)

        print("keyframe_objects = {}".format(keyframe_objects))
        print("frontier_objects = {}".format(frontier_objects))


        step_dict["keyframe_objects"] = keyframe_objects
        step_dict["keyframe_images"] = keyframe_images
        step_dict["frontier_objects"] = frontier_objects
        step_dict["frontier_images"] = frontier_images

        outputs, frontier_ids, keyframe_ids, reason = self.explore_step(
            step_dict, cfg, verbose=verbose
        )


        if outputs is None:
            logging.error(f"explore_step failed and returned None")
            return None
        logging.info(f"Response: [{outputs}]\nReason: [{reason}]")

        # 解析 vlm 给出的结果, 即目标类型和目标id
        # parse returned results
        try:
            target_type, target_index = outputs.split(" ")[0], outputs.split(" ")[1]
            logging.info(f"Prediction: {target_type}, {target_index}")
        except:
            logging.info(f"Wrong output format, failed!")
            return None

        # 目标类型为 snapshot 或者 frontier
        if target_type not in ["snapshot", "frontier"]:
            logging.info(f"Wrong target type: {target_type}, failed!")
            return None

        # 如果目标是snapshot, 则返回snapshot(是SnapShot类实例), ChatGPT给出的reason, 和prefiltering后剩余的snapshot的数量
        if target_type == "snapshot":
            if int(target_index) < 0 or int(target_index) >= len(keyframe_ids):
                logging.info(
                    f"Target index can not match real objects: {target_index}, failed!"
                )
                return None
            target_keyframe_id = keyframe_ids[int(target_index)]
            logging.info(f"The index of target snapshot {target_keyframe_id}")

            # get the target snapshot
            if target_keyframe_id not in scene.keyframes:
                logging.info(
                    f"Predicted snapshot target index is not in the map: {target_keyframe_id}, failed!"
                )
                return None

            pred_target_snapshot = scene.keyframes[target_keyframe_id]

            return pred_target_snapshot, reason
        # 如果目标是frontier, 则返回frontier(是Frontier类实例), ChatGPT给出的reason, 和prefiltering后剩余的snapshot的数量
        else:  # target_type == "frontier"
            target_frontier_id = frontier_ids[int(target_index)]
            target_frontier_index = frontier_id_to_index[target_frontier_id]
            if target_frontier_index < 0 or target_frontier_index >= len(tsdf_planner.frontiers):
                logging.info(
                    f"Predicted frontier target index out of range: {target_index}, failed!"
                )
                return None
            target_point = tsdf_planner.frontiers[target_frontier_index].position
            logging.info(f"Next choice: Frontier at {target_point}")
            pred_target_frontier = tsdf_planner.frontiers[target_frontier_index]

            return pred_target_frontier, reason