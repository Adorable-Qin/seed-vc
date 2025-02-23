import os
import json
from typing import Union
import soundfile as sf


with open("/mnt/workspace/home/fangzihao/WHSP_LGU/spk2uid.json", "r", encoding="utf-8") as f:
    spk2uid = json.load(f)
with open("/mnt/workspace/home/fangzihao/WHSP_LGU/uid2path.json", "r", encoding="utf-8") as f:
    uid2path = json.load(f)
with open("/mnt/workspace/home/fangzihao/WHSP_LGU/spk.json", "r", encoding="utf-8") as f:
    spk = json.load(f)

def spk2list(spk_: str, data_root: str = "/mnt/workspace/home/fangzihao/WHSP_LGU")-> Union[list, list]:
    norm_list = None
    whsp_list = [u for u in spk2uid[spk_] if "whsp" in u]
    whsp_list = [os.path.join(data_root, uid2path[u]) for u in whsp_list]
    if spk_ in spk["parallel"]:
        norm_list = [u for u in spk2uid[spk_] if "norm" in u]
        norm_list = [os.path.join(data_root, uid2path[u]) for u in norm_list]
    return whsp_list, norm_list
    
def conversion(knn_vc, src: str, ref: list):
    # use torch hub
    # knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    # use ckpt from github
    
    # Or, if you would like the vocoder trained not using prematched data, set prematched=False.
    src_wav_path = src
    ref_wav_paths = ref

    query_seq = knn_vc.get_features(src_wav_path)
    matching_set = knn_vc.get_matching_set(ref_wav_paths)

    out_wav = knn_vc.match(query_seq, matching_set, topk=4)
    # out_wav is (T,) tensor converted 16kHz output wav using k=4 for kNN.
    return out_wav
