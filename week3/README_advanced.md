## Q1) 어떤 task를 선택하셨나요?
> NER, MNLI, 기계 번역 셋 중 하나를 선택

# 1) MNLI 과제 링크
https://github.com/hongppa324/hhplus_ai/blob/main/week3/advanced_mnli.ipynb

# 2) NER 과제 링크
https://github.com/hongppa324/hhplus_ai/blob/main/week3/week3_ner_corrected.ipynb

# 3) 기계 번역 과제 링크 (학습 도중 시간이 부족해서 완료하지 못 했습니다.)
https://github.com/hongppa324/hhplus_ai/blob/main/week3/week3_translation_corrected.ipynb


## Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?
> 모델의 입력과 출력 형태 또는 shape을 정확하게 기술

# 1) MNLI
* 모델 : DistilBERT
* 입력 : Premise, Hypothesis 텍스트 
  - torch.Size([64, 400])
* 출력 : entailment, neutral, contradiction 중 하나
  - torch.Size([64, 3])

# 2) NER
* 모델 : DistilBertForTokenClassification
* 입력 : 토큰화된 단어
  - input_id, label : torch.Size([16, 128])
* 출력 : BIO tagging 형식의 tag
  - torch.Size([16, 128, 17])

# 3) 기계 번역

## Q3) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요? 
> 비교 metric은 loss curve, accuracy, 또는 test data에 대한 generalization 성능 등을 활용.
> +)이외에도 기계 번역 같은 문제에서 활용하는 BLEU 등의 metric을 마음껏 활용 가능
- 
-  
-  
- 이미지 첨부시 : ![이미지 설명](경로) / 예시: ![poster](./image.png)

### 위의 사항들을 구현하고 나온 결과들을 정리한 보고서를 README.md 형태로 업로드
### 코드 및 실행 결과는 jupyter notebook 형태로 같이 public github repository에 업로드하여 공유해주시면 됩니다. 반드시 출력 결과가 남아있어야 합니다.
