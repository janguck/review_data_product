## 소개

본 코드는 비꼬는 후기 감지와 과장, 허위광고 감지를 수행하는 코드입니다.

* 정의
    * `필수` : review_application_core
    * `부수` : review_application_web

## 환경

ubuntu 18.04, python 3.6.9, RTX 2080TI, requirements.txt


## review_application_core의 정의

### - generate_data.py : 과장, 허위광고 감지 기능에 대한 데이터를 생성하는 코드입니다. 

```python
cd review_application_core
python generate_data.py
```
 
 
### - train.py : 두 가지 기능에 대하여 모델이 학습을 진행하는 코드입니다.  

```python
cd review_application_core
python train.py [options]
```

* General Parameters:
    * `--model_name` (필수): 학습 할 기능 - (비꼬는 후기 감지-bert_sarcasm), (과장, 허위광고 감지-bert_siamese)
    * `--lr` : 학습시 러닝레이트
    * `--seq_len` : 리뷰, 방 설명에 대해 사용할 문자열의 길이 
    * `--gpu_id` : 사용할 그래픽카드 번호 
    * `--batch_size` : 학습시 배치 사이즈
    * `--epoch` : 학습될 횟수


### - common : 두 가지 기능에 대하여 공통적으로 사용되는 코드입니다. 

|  파일명        | 내용     | 
| :------------- | :---------- |
|  <strong>config.py</strong> | 기능, 그래픡카드, 배치사이즈와 같은 가변적으로 바뀔 수 있는 값을 입력 받는 코드입니다. | 
|  <strong>data_utils.py</strong> | 스파크, 데이터 샘플링와 같은 데이터와 관련된 함수들이 있는 코드입니다.   | 
|  <strong>file_utils.py</strong> | 학습된 모델에 대한 정보를 저장, 호출하는 코드입니다.   | 
|  <strong>metrics.py</strong> | 학습된 모델에 대하여 f1-score, precision, recall과 같은 지표를 측정하는 코드입니다.   | 
|  <strong>models.py</strong> | 버트 모델에 대하여 파인튜닝용 모델로 변환하는 코드입니다.   | 
|  <strong>tokenizer.py</strong> | 버트 모델에 학습되기 전 BPE, 인덱싱 해주는 코드입니다.    | 


### - data : 두 가지 기능에 대하여 모델이 학습, 예측하는 데이터입니다. 

|  폴더명        | 내용     | 
| :------------- | :---------- |
|  <strong>False_Exaggerated_advertisement.csv</strong> | 과장, 허위광고 감지 기능을 수행하기 위해 yelp 데이터를 과장, 허위광고 유/무에 대해 라벨링을 한 데이터입니다. | 
|  <strong>reddit</strong> | 비꼬는 후기 감지 기능에 사용되는 reddit 데이터입니다.    | 
|  <strong>yelp</strong> | 비꼬는 후기 감지 기능에 사용되는 yelp 데이터입니다.  | 


### - model : 두 가지 기능에 대하여 모델이 학습된 모델이 저장된 폴더입니다.

|  폴더명        | 내용     | 
| :------------- | :---------- |
|  <strong>FEAD</strong> | 과장, 허위광고 감지 모델 | 
|  <strong>SRD</strong> | 비꼬는 후기 감지 모델  | 


### - pretrain

|  폴더명        | 내용     | 
| :------------- | :---------- |
|  <strong>bert_base_uncased</strong> | 구글이 제공한 Bert-base 모델에 대한 단어장, 가중치가 있는 폴더입니다.| 


## review_application_web의 정의

**- django, mysql 사용합니다.**


### - django

1. application 폴더에 Service에 필요한 GET, Tokenizer, Database Save 코드가 있습니다.
2. Service 서버의 주소는 192.9.66.224 입니다.
3. Service의 기본적인 url은 application이며, 기능에 따라 뒤에 기능명이 붙습니다. ex) 192.9.66.224:8000/application/fead
4. FEAD, SRD 기능의 사용예시 코드는 sample_FEAD.py, sample_SRD.py 입니다.  


### - mysql

- FEAD

|  컬럼        | 내용     | 
| :------------- | :---------- |
|  <strong>description</strong> | 방에 대한 카테고리에 대한 설명 | 
|  <strong>review</strong> | 사용자의 리뷰 내용  | 
|  <strong>predict_label</strong> | 본 모델이 예측한 라벨  | 


- SRD

|  컬럼        | 내용     | 
| :------------- | :---------- |
|  <strong>review</strong> | 사용자의 리뷰 내용  | 
|  <strong>predict_label</strong> | 본 모델이 예측한 라벨  | 


## 성능

### 허위, 과장광고 감지 (yelp)

|  모델        | Accuracy     |  F1-score        |
| :------------- | :---------- | :---------- |
|  <strong>BERT-siamese</strong> | 0.77   | 0.78  | 
|  <strong>Text Similarity Using Siamese Deep Neural Network(Mueller, 2016)</strong> | 0.62   |  0.59   |
|  <strong>One Label Prediction</strong> | 0.66   |  -   |

### 비꼬는 후기 감지(yelp, reddit)

|  모델        | Accuracy     |  F1-score        |
| :------------- | :---------- | :---------- |
|  <strong>BERT-sarcasm</strong> | 0.74   | 0.76  | 
|  <strong>SarcasmDetection(Khodak, 2017)</strong> | 0.68   |  0.68   |
|  <strong>One Label Prediction</strong> | 0.51   |  -   |


## 참고
1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(arXiv preprint)
2. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
3. Siamese Recurrent Architectures for Learning Sentence Similarity(AAAI2016)
4. A Large Self-Annotated Corpus for Sarcasm(Proceedings of the Linguistic Resource and Evaluation Conference)
