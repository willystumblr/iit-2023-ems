# `pv` 디렉토리 설명

## Files

```plaintext
.
├── \010ict_2023_model_predict.ipynb 	### Train, validate, test notebook (Google Colaboratory)
├── README.md 
├── eda.ipynb 				### Data EDA
├── fix.py				### Data fixation; fillna, ffill, etc.
├── preprocess.py
├── checkpoints				### Model checkpoints for each building
│   ├── LG도서관-checkpoint-2023-08-10 22_10.pt
│   ├── 대학C동-checkpoint-2023-08-10 22_35.pt
│   ├── ...
│   ├── 중앙도서관-checkpoint-2023-08-10 23_26.pt
│   └── 신재생에너지동-checkpoint-2023-08-10 23_18.pt
├── eval_sequence			### each building's evaluation sequeunce; for predicting 08/31/2022
│   ├── LG도서관-eval.csv
│   ├── 대학C동-eval.csv
│   ├── ...
│   ├── 삼성환경동-eval.csv
│   └── 신재생에너지동-eval.csv
└── results				### storing hyperparameters; num_layer, batch_size, etc.
    ├── LG도서관-2023-08-11-hp.pkl
    ├── ...
    ├── 중앙도서관-2023-08-11-hp.pkl
    └── 신재생에너지동-2023-08-11-hp.pkl
```

## 파일 개별 설명

- `\010ict_2023_model_predict.ipynb` : Google Colaboratory notebook, 모델 훈련 및 검증까지 진행
- `eda.ipynb` : 주어진 데이터셋에 NaN 값, 정상범위를 현저히 넘어선 값 등이 존재하는지 확인
- `preprocess.py` : 분석을 거친 데이터에서, 불필요한 column 제거 및 비정상적인 값, 비어있는 값 수정
- `fix.py` : `preprocess.py`가 사용하는 함수가 저장되어 있는 파일
- `checkpoints/` : 건물별 모델 체크포인트
- `eval_sequence/` : 2022년 8월 31일 예측을 위해 LSTM 모델이 사용하는 시퀀스를 따로 저장함
- `results/` : 가장 성능이 좋았던 모델의 hyperparameter를 pickle 객체로 저장

# Results

각 건물별 test 과정에서, 정규화된 RSME 값은 다음과 같다.

```plaintext
[LG도서관] Test Loss (RSME): 0.21458798088133335
[기숙사B동] Test Loss (RSME): 0.22306846295084273
[다산빌딩] Test Loss (RSME): 0.21325922012329102
[대학C동] Test Loss (RSME): 0.2414388805627823
[동물실험동] Test Loss (RSME): 0.2286185510456562
[산업협력관] Test Loss (RSME): 0.22067535562174662
[삼성환경동] Test Loss (RSME): 0.228206068277359
[시설관리동] Test Loss (RSME): 0.21965214982628822
[신재생에너지동] Test Loss (RSME): 0.22857287526130676
[중앙도서관] Test Loss (RSME): 0.21714665926992893
[중앙창고] Test Loss (RSME): 0.21500113606452942
[축구장] Test Loss (RSME): 0.13879267312586308
[학사과정] Test Loss (RSME): 0.1885223723948002
[학생회관] Test Loss (RSME): 0.13863694667816162
```

Google Colaboratory T4 GPU 1장으로 예측을 진행했을 때, 2022/08/31의 실제 값과 오차는 다음과 같았다.

```plaintext
Total Error for this config: 0.5441899299621582kWh
```

위 결과는 `\010ict_2023_model_predict.ipynb` 에서 cell output으로 확인된다.
