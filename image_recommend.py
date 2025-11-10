import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel
from sklearn.metrics.pairwise import cosine_similarity
import shutil

# 检查是否有 CUDA 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练模型
processor = AutoImageProcessor.from_pretrained("./siglip2")
model = SiglipVisionModel.from_pretrained("./siglip2").to(device).eval()


def find_similar_images(target_image_path, embeddings_matrix, image_paths, top_k=10, output_folder="./similar_images"):
    """
    给定目标图像路径，计算其与所有图像的相似度，选择最相似的 top_k 张图像，并保存。
    """
    file_name = os.path.basename(target_image_path).split(".")[0]
    output_folder = os.path.join(output_folder, file_name)

    # 读取目标图像
    target_image = Image.open(target_image_path).convert("RGB")
    target_inputs = processor(images=target_image, return_tensors="pt").to(device)
    with torch.no_grad():
        target_outputs = model(**target_inputs)
        target_emb = target_outputs.pooler_output
        target_emb = torch.nn.functional.normalize(target_emb, dim=-1).cpu().numpy()

    # 计算目标图像与所有图像的相似度
    similarities = cosine_similarity(target_emb, embeddings_matrix)
    print(similarities)
    
    # 获取最相似的 top_k 图像索引
    top_k_indices = similarities[0].argsort()[-top_k:][::-1]
    print(top_k_indices)
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取相似图像并保存
    for idx in top_k_indices:
        img_path = image_paths[idx]  # 从编码矩阵中获取对应的图像路径
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_folder, f"{img_name}")
        shutil.copy(img_path, output_path)  # 将最相似的图像复制到新的文件夹
        print(f"Saved similar image: {img_name} to {output_folder}")

# 示例：读取之前保存的图像编码，进行相似度计算
embeddings_matrix = np.load("/mnt/vdb2t_1/sujunyan/program/ui/image_features_extract/embedding/clean_embeddings.npy")
path_matrix = np.load("/mnt/vdb2t_1/sujunyan/program/ui/image_features_extract/embedding/clean_paths.npy")

# 运行相似度计算
find_similar_images("/mnt/vdb2t_1/sujunyan/label30000/Pattern_recognition_filter/7000_results/final_good/--sjYu-rjCJ2DoJLcjEkUg_2025-09-24_7_848_f_7_848_bbox2.png", 
                    embeddings_matrix, image_paths=path_matrix, top_k=10, output_folder="./results/qvchong")
