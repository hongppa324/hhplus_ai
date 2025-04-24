# Q1) 어떤 task를 선택하셨나요?
> 주어진 예시 중 논문 요약하는 task 선택

## 과제 링크
https://github.com/hongppa324/hhplus_ai/blob/main/week5/advanced_paper.ipynb

## 과제 요약
### [1] PDF를 load하는 라이브러리를 사용

### [2] custom_prompt 설정
#### 1. 논문을 요약 전문가로 설정
#### 2. 요약에 포함할 내용
##### 1) 논문의 의미
##### 2) 논문의 method나 접근 방법
##### 3) 주요 발견 및 결과
##### 4) 논문의 인사이트
#### 3. 문맥에 없는 사실을 만들어내지 말 것
#### 4. reference나 인용은 포함시키지 말 것
#### 5. 명확하고 학술적인 언어로 작성할 것

### 요약 결과
#### 1) RAG를 도입한 모델 설명
#### 2) method : retrieval mechanism을 이용한 답변 생성
#### 3) 결과 : RAG를 이용하여 독해 benchmark에서 뛰어난 성능을 보임
#### 4) insight : 복잡도를 크게 증가시키지 않고도 자연어 처리를 잘 할 수 있음을 시사

The paper presents a novel approach to multi-paragraph reading comprehension by introducing the RAG (Retrieval-Augmented Generation) model, which effectively combines parametric and non-parametric memory components to improve performance in tasks requiring contextual understanding. The proposed methodology utilizes retrieval mechanisms to enhance the generation of responses based on external knowledge sources, allowing for better contextual relevance and accuracy. The key findings demonstrate that RAG achieves competitive results in reading comprehension benchmarks, outperforming traditional models without needing complex pipeline architectures or extensive retrieval supervision. Notably, the work highlights the potential of integrating retrieval methods into generative models, which can lead to advances in various natural language processing applications by leveraging external knowledge efficiently without significant increases in complexity.