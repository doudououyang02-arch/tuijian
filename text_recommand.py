rom FlagEmbedding import BGEM3FlagModel
import numpy as np
from sentence_transformers import util
import torch
import shutil
import os

m3 = BGEM3FlagModel("/mnt/vdb2t_1/sujunyan/program/ui/image_features_extract/models/baai_bge-m3/baai_bg3-m3", use_fp16=False)  # 1024 维，最长 8192 tokens


# 保存到本地
text_embeddings_matrix = np.load("text_embeddings_8820.npy")
    

path_matrix = np.load("path_matrix_8820.npy")


def cosine(a, b):  # a:[d], b:[N,d]
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (b @ a).reshape(-1)

def lexical_score(q_ws, d_ws):
    # 官方给了 compute_lexical_matching_score；这里示意手工版本
    s = 0.0
    for tok, w in q_ws.items():
        if tok in d_ws: s += w * d_ws[tok]
    return s

def colbert_maxsim(q_vecs, d_vecs):
    # 官方提供 model.colbert_score；这里给一个近似实现
    import numpy as np
    qn = q_vecs / (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-9)
    dn = d_vecs / (np.linalg.norm(d_vecs, axis=1, keepdims=True) + 1e-9)
    return (qn @ dn.T).max(axis=1).mean()

query = "4行5列的表格"

q_out = m3.encode(
    [query],
    return_dense=True, return_sparse=True, return_colbert_vecs=True
)
q_dense = q_out["dense_vecs"][0]
q_ws    = q_out["lexical_weights"][0]
q_cols  = q_out["colbert_vecs"][0]

# 3. 计算相似度（矩阵化）
# 稠密向量的余弦相似度
dense_sim = util.cos_sim(q_dense, text_embeddings_matrix)  # [1, N]
dense_sim = dense_sim.reshape(-1)  # 扁平化为 1D

scores = dense_sim
print(scores)

# 5. 按分数排序获取 Top-K 候选
topk = torch.argsort(scores,descending=True)[:10]
candidates = [(scores[i], path_matrix[i]) for i in topk]

# 输出 Top-K
for score, text in candidates:
    print(f"Score: {score:.4f} | Text: {text}")
    
    # img_path = image_paths[idx]  # 从编码矩阵中获取对应的图像路径
    img_name = os.path.basename(text)

    json_path = os.path.join("/".join(text.split("/")[:-1]).replace("final_good", "final_good_json"), img_name.replace(".png", ".json"))
    # print(json_path)
    # exit()
    output_path = os.path.join("./text_results", f"{img_name}")
    output_json_path = os.path.join("./text_results", f"{os.path.basename(json_path)}")
    os.makedirs("./text_results", exist_ok=True)
    shutil.copy(text, output_path)  # 将最相似的图像复制到新的文件夹
    shutil.copy(json_path, output_json_path)  # 将最相似的图像复制到新的文件夹
    print(f"Saved similar image: {img_name} to {output_path}")
