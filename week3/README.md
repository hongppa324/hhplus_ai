### 과제 요구 사항들을 구현하고, epoch마다의 train loss와 최종 모델의 test accuracy가 print된 notebook을 public github repository에 업로드하여 공유해주시면 됩니다. 반드시 출력 결과가 남아있어야 합니다. 
- [ ] AG_News dataset 준비
	- Huggingface dataset의 `fancyzhx/ag_news`를 load
	- `collate_fn` 함수에 다음 수정사항들을 반영
    - Truncation과 관련된 부분들을 삭제
- [ ] Classifier output, loss function, accuracy function 변경
	- 뉴스 기사 분류 문제는 binary classification이 아닌 일반적인 classification 문제입니다. MNIST 과제에서 했던 것 처럼 `nn.CrossEntropyLoss` 를 추가하고 `TextClassifier`의 출력 차원을 잘 조정하여 task를 풀 수 있도록 수정
	- 그리고 정확도를 재는 `accuracy` 함수도 classification에 맞춰 수정
- [ ]  학습 결과 report
    - DistilBERT 실습과 같이 매 epoch 마다의 train loss를 출력하고 최종 모델의 test accuracy를 report 첨부
  
- [x] 완료된 항목은 체크박스에 x 표시

- 이미지 첨부시 : ![이미지 설명](경로) / 예시: ![poster](./image.png)
