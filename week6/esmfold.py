from transformers import EsmForProteinFolding, AutoTokenizer
from Bio import SeqIO
import torch
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# FASTA 파일에서 sequence 읽기
fasta_path = "./4CSV.fasta"
record = next(SeqIO.parse(fasta_path, "fasta"))
sequence = str(record.seq)

valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWYBXZOU")  # U: Selenocysteine 허용
invalid_chars = [aa for aa in sequence if aa not in valid_amino_acids]

if invalid_chars:
    print("허용되지 않는 문자 발견 ==> ", set(invalid_chars))
else:
    print("모든 문자가 유효합니다.")

# print(f"FASTA에서 읽은 sequence: {sequence}, {len(sequence)}")

# Tokenizer와 모델 로드
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
# print("tokenizer", tokenizer)
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").cuda()
# print("model", model)

# 토큰화
inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
inputs = {k: v.cuda() for k, v in inputs.items()}
inputs.pop("token_type_ids", None)

# 모델 호출
with torch.no_grad():
    outputs = model(**inputs, num_recycles=1)

# PDB 문자열 추출 및 저장
pdb_str = outputs["pdb"]
with open("predicted_structure.pdb", "w") as f:
    f.write(pdb_str)

print("PDB 파일 저장 완료: predicted_structure.pdb")