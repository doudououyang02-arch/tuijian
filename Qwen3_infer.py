import numpy as np
from sentence_transformers import util
import torch
import shutil
import os
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import json
tokenizer = AutoTokenizer.from_pretrained('/home/s50052424/UIRecommend/Qwen3-Embedding-8B/', padding_side='left')
model = AutoModel.from_pretrained('/home/s50052424/UIRecommend/Qwen3-Embedding-8B/', dtype=torch.bfloat16).cuda().eval()

# 保存到本地
text_embeddings_matrix = torch.tensor(np.load("text_embeddings_8820.npy")).cuda()
path_matrix = np.load("path_matrix_8820.npy")

    
def func(semantics, return_dense=True, return_sparse=True, return_colbert_vecs=True, max_length=8192):
    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    batch_dict = tokenizer(
        semantics,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict.to(model.device)
    with torch.no_grad():
        outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return {"dense_vecs": embeddings, "lexical_weights":[None], "colbert_vecs":[None]}

model.encode = func

def query_top(query):
    with torch.no_grad():
        q_out = model.encode(
            [query],
            return_dense=True, return_sparse=True, return_colbert_vecs=True
        )
    q_dense = q_out["dense_vecs"][0]
    q_ws    = q_out["lexical_weights"][0]
    q_cols  = q_out["colbert_vecs"][0]

    # 3. 计算相似度（矩阵化）
    # 稠密向量的余弦相似度
    with torch.no_grad():
        dense_sim = util.cos_sim((q_dense).float(), text_embeddings_matrix)  # [1, N]
    dense_sim = dense_sim.reshape(-1)  # 扁平化为 1D

    scores = dense_sim
    print(scores)

    # 5. 按分数排序获取 Top-K 候选
    top_k = torch.argsort(scores,descending=True)[:50]
    candidates = [(scores[i], path_matrix[i]) for i in top_k]

    # 输出 Top-K
    for score, text in candidates:
        text = text.replace("mnt/vdb2t_1/sujunyan/label30000", "home/s50052424/UIRecommend")
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
    return top_k

query = "4行5列的表格"
top_k = query_top(query)

