import openai
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging
from const import *


client = OpenAI(
    base_url=END_POINT,
    api_key=OPENAI_KEY,
)


def format_content(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{c[1]}",
                        "detail": "high",
                    },
                }
            )
    return formated_content


# send information to openai
def call_openai_api(sys_prompt, contents) -> Optional[str]:
    '''
    调用OpenAI API, 获取回复
    '''
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]
    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",  # model = "deployment_name"
                messages=message_text,
                temperature=0.7,
                max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 60s")
            time.sleep(30)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(60)
            retry_count += 1
            continue

    return None


# encode tensor images to base64 format
def encode_tensor2base64(img):
    '''
    把numpy图像编码成 base64 格式
    '''
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


def format_question(step):
    question = step["question"]
    image_goal = None
    if "task_type" in step and step["task_type"] == "image":
        with open(step["image"], "rb") as image_file:
            image_goal = base64.b64encode(image_file.read()).decode("utf-8")

    return question, image_goal


def get_step_info(step, verbose=False):
    '''
    整理query所需的信息, 编码数据, prefoltering. 返回值:
        question: 问题
        image_goal: 如果问题类型是image, 则会包含一个图片
        egocentric_imgs: 各个step每个视角的图片(resize过的)
        frontier_imgs: frontier 对应的 resize后的图片
        snapshot_imgs: 各个snapshot对应的resize后的图片
        snapshot_classes: snapshot中(也是地图中)包含的所有的物体的类别
        keep_index: 包含 prefiltering中被选中和question相关的物体 的snapshot 的 index列表
    '''

    # 获取问题数据, question是文本问题, image_goal是base64格式的图像数据
    # 只有task_type是image时, 才会有image_goal
    # 1 get question data
    question, image_goal = format_question(step)

    # 
    # 1 prefiltering, note that we need the obj_id_mapping
    if step.get("use_prefiltering") is True:
        class_labels = set(step["keyframe_objects"].keys()) | set(step["frontier_objects"].keys())
        if len(class_labels) > step["top_k_categories"] and (len(step["keyframe_images"]) + len(step["frontier_images"])) > 10:
            # 论文中的prefiltering, 这时的snapshot_classes里只包含ChatGPT推测的和question相关的object
            keep_classes = prefiltering(
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
                frontier_imgs.append(encode_tensor2base64(step["frontier_images"][frontier_id]))
                frontier_ids.append(frontier_id)
                frontier_added_ids.add(frontier_id)


    keyframe_imgs, keyframe_ids, keyframe_added_ids  = [], [], set()
    for class_label, keyframe_id_set in step["keyframe_objects"].items():
        for keyframe_id in keyframe_id_set:
            if keyframe_id not in keyframe_added_ids:
                keyframe_imgs.append(encode_tensor2base64(step["keyframe_images"][keyframe_id]))
                keyframe_ids.append(keyframe_id)
                keyframe_added_ids.add(keyframe_id)

    print("keep keyframe = {}, keep frontier = {}".format(len(keyframe_added_ids), len(frontier_added_ids)))

    return question, image_goal, frontier_imgs, frontier_ids, keyframe_imgs, keyframe_ids


def format_explore_prompt(
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
    text = f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    text = "Select the Frontier/Snapshot that would help find the answer of the question.\n"
    content.append((text,))



    # 3 here is the snapshot images
    # text = "The followings are all the snapshots that you can choose (followed with contained object classes)\n"
    # text += "Please note that the contained classes may not be accurate (wrong classes/missing classes) due to the limitation of the object detection model. "
    # text += "So you still need to utilize the images to make decisions.\n"
    text = "The followings are all the snapshots that you can choose\n"
    content.append((text,))
    if len(keyframe_imgs) == 0:
        content.append(("No Snapshot is available\n",))
    else:
        for i in range(len(keyframe_imgs)):
            content.append((f"Snapshot {i} ", keyframe_imgs[i]))
            # if use_snapshot_class:
            #     text = ", ".join(snapshot_classes[i])
            #     content.append((text,))
            content.append(("\n",))

    # 4 here is the frontier images
    text = "The followings are all the Frontiers that you can explore: \n"
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available\n",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"Frontier {i} ", frontier_imgs[i]))
            content.append(("\n",))

    # 5 here is the format of the answer
    text = "Please provide your answer in the following format: 'Snapshot i\n[Answer]' or 'Frontier i\n[Reason]', where i is the index of the snapshot or frontier you choose. "
    text += "For example, if you choose the first snapshot, you can return 'Snapshot 0\nThe fruit bowl is on the kitchen counter.'. "
    text += "If you choose the second frontier, you can return 'Frontier 1\nI see a door that may lead to the living room.'.\n"
    text += "Note that if you choose a snapshot to answer the question, (1) you should give a direct answer that can be understood by others. Don't mention words like 'snapshot', 'on the left of the image', etc; "
    text += "(2) you can also utilize other snapshots, frontiers and egocentric views to gather more information, but you should always choose one most relevant snapshot to answer the question.\n"
    content.append((text,))

    return sys_prompt, content


def format_prefiltering_prompt(question, class_list, top_k=10, image_goal=None):
    content = []
    sys_prompt = "You are an AI agent in a 3D indoor scene.\n"
    prompt = "Your goal is to answer questions about the scene through exploration.\n"
    prompt += "To efficiently solve the problem, you should first rank objects in the scene based on their importance.\n"
    prompt += "These are the rules for the task.\n"
    prompt += "1. Read through the whole object list.\n"
    prompt += "2. Rank objects in the list based on how well they can help your exploration given the question.\n"
    prompt += f"3. Reprint the name of all objects that may help your exploration given the question. "
    prompt += "4. Do not print any object not included in the list or include any additional information in your response.\n"
    content.append((prompt,))
    # ------------------format an example-------------------------
    prompt = "Here is an example of selecting helpful objects:\n"
    prompt += "Question: What can I use to watch my favorite shows and movies?\n"
    prompt += (
        "Following is a list of objects that you can choose, each object one line\n"
    )
    prompt += "painting\nspeaker\nbox\ncabinet\nlamp\ntv\nbook rack\nsofa\noven\nbed\ncurtain\n"
    prompt += "Answer: tv\nspeaker\nsofa\nbed\n"
    content.append((prompt,))
    # ------------------Task to solve----------------------------
    prompt = f"Following is the concrete content of the task and you should retrieve helpful objects in order:\n"
    prompt += f"Question: {question}"
    if image_goal is not None:
        content.append((prompt, image_goal))
        content.append(("\n",))
    else:
        content.append((prompt + "\n",))
    prompt = (
        "Following is a list of objects that you can choose, each object one line\n"
    )
    for i, cls in enumerate(class_list):
        prompt += f"{cls}\n"
    prompt += "Answer: "
    content.append((prompt,))
    return sys_prompt, content


def get_prefiltering_classes(question, seen_classes, top_k=10, image_goal=None):
    '''
    构建prompt, 获取OpenAI回复, 并从回复中抽取 与本次question相关的object种类
    '''
    # 构建 prompt, 返回值分别是系统prompt, 和当前问题的prompt
    prefiltering_sys, prefiltering_content = format_prefiltering_prompt(
        question, sorted(list(seen_classes)), top_k=top_k, image_goal=image_goal
    )

    # 调OpenAI API, 获取 prefiltering 回复
    message = ""
    for c in prefiltering_content:
        message += c[0]
        if len(c) == 2:
            message += f": image {c[1][:10]}..."
    response = call_openai_api(prefiltering_sys, prefiltering_content)
    if response is None:
        return []

    # 后处理OpenAI回复, 抽取相关物体
    # parse the response and return the top_k objects
    selected_classes = response.strip().split("\n")
    selected_classes = [cls.strip() for cls in selected_classes]
    selected_classes = [cls for cls in selected_classes if cls in seen_classes]
    selected_classes = selected_classes[:top_k]

    return selected_classes


def prefiltering(
    question, seen_classes, top_k=10, image_goal=None, verbose=False
):
    '''
    论文中的prefiltering, 返回snapshot_classes, keep_index
    snapshot_classes: 列表, 每个元素是一个snapshot包含的物体种类, 这时的snapshot_classes里只包含ChatGPT推测的和question相关的object
    keep_index: 包含 selected_classes 物体的 snapshot 的 index列表
    '''
    # 构建prompt, 获取OpenAI回复, 并从回复中抽取 与本次question相关的object种类
    selected_classes = get_prefiltering_classes(
        question, seen_classes, top_k, image_goal
    )
    if verbose:
        logging.info(f"Prefiltering selected classes: {selected_classes}")


    return selected_classes


def explore_step(step, cfg, verbose=False):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories

    question, image_goal, frontier_imgs, frontier_ids, keyframe_imgs, keyframe_ids = get_step_info(step, verbose)


    # 整理prompt
    sys_prompt, content = format_explore_prompt(question, frontier_imgs, keyframe_imgs, image_goal=image_goal)

    if verbose:
        logging.info(f"Input prompt:")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    retry_bound = 3
    final_response = None
    final_reason = None
    for _ in range(retry_bound):
        full_response = call_openai_api(sys_prompt, content)

        if full_response is None:
            print("call_openai_api returns None, retrying")
            continue

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
            continue

        response_valid = False
        if (
            choice_type == "snapshot"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(keyframe_imgs)
        ):
            response_valid = True
        elif (
            choice_type == "frontier"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(frontier_imgs)
        ):
            response_valid = True

        if response_valid:
            final_response = response
            final_reason = reason
            break

    return final_response, frontier_ids, keyframe_ids, final_reason
