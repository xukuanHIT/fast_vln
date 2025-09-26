import rerun as rr
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import textwrap
from PIL import Image, ImageDraw, ImageFont
import cv2


def concat_images_opencv(img_list, gap=10, bg_color=(255,255,255)):
    n = len(img_list)
    W = img_list[0].shape[1]  
    cols = 2
    rows = (n + 1) // 2

    row_heights = []
    for r in range(rows):
        imgs = img_list[r*cols:(r+1)*cols]
        h = max(im.shape[0] for im in imgs)
        row_heights.append(h)

    total_height = sum(row_heights) + (rows-1)*gap
    total_width = cols*W + (cols-1)*gap

    canvas = np.full((total_height, total_width, 3), bg_color, dtype=np.uint8)

    y_offset = 0
    for r in range(rows):
        imgs = img_list[r*cols:(r+1)*cols]
        x_offset = 0
        for im in imgs:
            h, w = im.shape[:2]
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = im
            x_offset += w + gap
        y_offset += row_heights[r] + gap

    return canvas



class Visualization:
    def __init__(self, cfg,):

        rr.init("visualization", spawn=True)
        # rr.connect()
        rr.log("world", rr.Transform3D())

        self.cmap = plt.get_cmap("tab20")

        self.object_ids = set()
        self.trajectory_points = []

    def o3d_color_to_uint8(self, colors):
        """Open3D colors usually in float [0,1] -> uint8 [0,255]"""
        cols = np.asarray(colors)
        return (np.clip(cols, 0.0, 1.0) * 255).astype(np.uint8)


    def update_image(self, image):
        rr.log("camera/image", rr.Image(image))


    def add_trajectory_point(self, position):
        # rr.log("world/trajectory", rr.Clear(recursive=True))
        self.trajectory_points.append(position)
        rr.log("world/trajectory", rr.LineStrips3D([np.array(self.trajectory_points)], radii=0.02))


    def delete_trajectory(self):
        self.trajectory_points.clear()



    def add_object(self, object_id, pcd):
        if object_id in self.object_ids:
            self.delete_object(object_id)

        pts = np.asarray(pcd.points)
        if pts.size == 0:
            return 

        # colors: 如果点云自带颜色就用它，否则用单色（来自 colormap）
        # if pcd.has_colors():
        if False:
            colors = self.o3d_color_to_uint8(np.asarray(pcd.colors))
        else:
            base_color = np.array(self.cmap(object_id % 20)[:3])  # float 0..1
            colors = (np.tile(base_color, (len(pts), 1)) * 255).astype(np.uint8)

        # 建议的路径层级： world/objects/object_{i}
        base_path = f"world/objects/object_{object_id}"

        # （可选）把物体的坐标系也写入（这里用单位位姿）
        rr.log(f"{base_path}/pose", rr.Transform3D())

        # 写入点云（Points3D 接受 points, colors）
        rr.log(f"{base_path}/points", rr.Points3D(pts, colors=colors))

        # # 如果想可视化每个物体的质心为一个小球/点
        # centroid = pts.mean(axis=0)
        # rr.log(f"{base_path}/centroid", rr.Points3D(np.array([centroid]), radii=0.03))

        self.object_ids.add(object_id)


    def delete_object(self, object_id):
        if object_id in self.object_ids:
            base_path = f"world/objects/object_{object_id}"
            rr.log(base_path, rr.Clear(recursive=True))
            self.object_ids.discard(object_id)


    def update_object(self, object_id, pcd):
        if object_id in self.object_ids:
            self.delete_object(object_id)
        
        self.add_object(object_id, pcd)

    def update_chat(self, text_lines):
        dialog_text = ""
        # for i, line in enumerate(text_lines):
        #     if i == 0:
        #         dialog_text += f"<span style='color:red; font-weight:bold'>{line}</span><br>\n"
        #     else:
        #         dialog_text += f"<span style='color:blue'>{line}</span><br>\n"

        dialog_text = f"**{text_lines[0]}\n\n" + "\n".join(f"{line}" for line in text_lines[1:])

        rr.log("world/dialog", rr.TextDocument(dialog_text))