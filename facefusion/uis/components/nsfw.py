from typing import Optional, List
import gradio
from transformers import pipeline
from PIL import Image
import io
import cv2
import os
import face_recognition
import numpy as np

from facefusion.filesystem import is_image, is_video
from facefusion.uis.typing import File

COMMON_NSFW_FILE: Optional[gradio.File] = None
COMMON_NSFW_RESULT: Optional[gradio.JSON] = None

SOURCE_WIDTH = None
SOURCE_HEIGHT = None


def render() -> None:
	global COMMON_NSFW_FILE
	global COMMON_NSFW_RESULT

	COMMON_NSFW_FILE = gradio.File(label="鉴黄")
	COMMON_NSFW_RESULT = gradio.JSON(label="鉴黄结果")


def listen() -> None:
	COMMON_NSFW_FILE.change(nsfw, inputs=COMMON_NSFW_FILE, outputs=COMMON_NSFW_RESULT)


def nsfw(file: File):
	if file:
		if is_video(file.name):
			return nsfw_video(file)
		if is_image(file.name):
			return nsfw_image(file)


# 初始化模型
classifier = pipeline('image-classification', model="/root/autodl-fs/face_fusion/model/Falconsai/nsfw_image_detection")
reference_images_dir="/root/autodl-fs/face_fusion/face"

#鉴黄评分控制(越高越黄)
score_threshold = 0.8
# 人脸识别设置阈值(越低越接近)
face_threshold = 0.5

def classify_image(pil_image):
	# 对单个图像进行分类
	return classifier(pil_image)


def nsfw_image(file):
	global SOURCE_WIDTH
	global SOURCE_HEIGHT
	# 检测图片中是否存在NSFW内容
	try:
		with Image.open(file.name) as image:
			# 对图片进行分类处理
			results = classify_image(image)

			SOURCE_WIDTH,SOURCE_HEIGHT = image.size
			# 根据分类结果判断是否为NSFW
			for result in results:
				if result['label'] == 'nsfw' and result['score'] > score_threshold:
					return {"code": 201, "msg": "检测到敏感内容"}
		if same_face([image]):
			return {"code": 201, "msg": "检测到既定人脸"}
	except Exception as e:
		return {"code": 500, "msg": str(e)}

	return {"code": 200, "msg": "", "data": {"width": SOURCE_WIDTH, "height": SOURCE_HEIGHT, "is_image": True, "is_video": False}}


def nsfw_video(file):
	# 检测视频中是否存在NSFW内容
	try:
		for frame in frame_generator(file.name):
			pil_image = Image.fromarray(frame)
			is_nsfw = classify_image(pil_image)
			for result in is_nsfw:
				if result['label'] == 'nsfw' and result['score'] > score_threshold:
					return {"code": 202, "msg": "检测到敏感内容"}
			if same_face([pil_image]):
				return {"code": 201, "msg": "检测到既定人脸"}
	except Exception as e:
		return {"code": 500, "msg": str(e)}
	return {"code": 200, "msg": "", "data": {"width": SOURCE_WIDTH, "height": SOURCE_HEIGHT, "is_image": False, "is_video": True}}

def frame_generator(video_path):
	global SOURCE_WIDTH
	global SOURCE_HEIGHT

	video = cv2.VideoCapture(video_path)
	SOURCE_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	SOURCE_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	# 获取视频的FPS
	fps = video.get(cv2.CAP_PROP_FPS)
	# 根据视频的FPS来计算间隔多少帧提取一帧
	frame_interval = int(round(fps))

	# 初始化帧计数器
	frame_counter = 0

	while video.isOpened():
		ret, frame = video.read()
		if not ret:
			break

		# 每秒提取一帧
		if frame_counter % frame_interval == 0:
			yield frame

		frame_counter += 1

	video.release()

def load_and_encode_faces(filenames, directory):
	# 用于存储人脸编码
	faces_encodings = []
	# 加载并编码图片
	for filename in filenames:
		image_path = os.path.join(directory, filename)
		image = face_recognition.load_image_file(image_path)
		# 假定每张图片包含一张人脸，取第一个编码
		face_encodings = face_recognition.face_encodings(image)
		if face_encodings:
			faces_encodings.extend(face_encodings)
	return faces_encodings

def calculate_face_distance(face_encoding1, face_encoding2):
	# 计算两个人脸编码之间的欧氏距离
	return np.linalg.norm(face_encoding1 - face_encoding2)

def same_face(sample_image_sources):
	# 加载参考图片，并对每张图片进行编码
	reference_images_filenames = os.listdir(reference_images_dir)
	reference_faces_encodings = load_and_encode_faces(reference_images_filenames, reference_images_dir)

	# 对示例图片进行编码
	sample_faces_encodings = []
	for sample_image_source in sample_image_sources:
		if sample_image_source.mode != 'RGB':
			sample_image_source = sample_image_source.convert('RGB')
		np_image = np.array(sample_image_source)
		encodings = face_recognition.face_encodings(np_image)
		sample_faces_encodings.extend(encodings)

	# 检查是否有匹配的人脸
	for sample_face_encoding in sample_faces_encodings:
		# 计算当前人脸编码与参考组中每个编码的距离，并检查是否有距离小于阈值的
		distances = [calculate_face_distance(sample_face_encoding, face_encoding) for face_encoding in reference_faces_encodings]
		if any(distance < face_threshold for distance in distances):
			return True

	return False
