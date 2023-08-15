## Methods

 시계열 데이터 학습을 위해 LSTM 모델을 사용하였으며, regularization 및 학습 효율성을 위해 Early Stopping method를 적용했기 때문에 epoch 수는 충분히 크도록 설정하면 된다. Optimizer로는 pytorch에서 제공하는 Adam을 사용했으며, 추가적으로 ReduceLROnPlateau scheduler를 적용했다. 이는 validation loss가 일정 epoch동안 감소하지 않을 경우 learning rate를 동적으로 감소시키게 된다. Early stopping과 동시에 사용하기 위해 patience 파라미터는 실험적으로 설정했고, 따라서 변경하지 않는 것이 좋다. 해당 reference는 아래 링크를 참고한다.

* https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
* https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html

 주어진 데이터셋은 train set, validation set, test set으로 구분하였으며 7:2:1 비율의 시퀀스별로 나누었다. 주어진 전력사용량 데이터셋을 확인해보면 총 56개의 건물에서 '산학협력연구동'과 '산학협력연구동(E)를 제외하고 54개의 건물이 MWH 단위의 유효전력량을 가지고 있는데, 문의한 결과 KWH로 되어있는 해당 2개의 건물도 MWH 단위가 맞다고 한다. 전력요금을 계산하기 위해서는 유효전력량(MWH)을 알아야하지만, 대회에서 제공한 데이터셋을 확인해본 결과 해당 값은 지금까지의 누적 값이며 천의 자리수에서 0으로 초기화되는 특성이 있었다. 따라서 본 모델 개발 과정에서는 유효전력(KW)를 예측하며, 이를 이용해 유효전력량 및 전기요금을 계산하도록 구현하였다. 또한 데이터셋은 학사 구역과 석사 구역으로 나뉘어져 있으며 SV-1, SV-2 등등 동일한 이름의 건물들이 존재했는데, 이는 모두 다른 건물로 취급한다.

## 각 파일에 대한 설명

```
├── load
│   ├── log_optuna.txt : 하이퍼파라미터 최적화 로그
│   ├── main.ipynb : 모델 구현 및 학습, 평가 코드
│   ├── pickleviewer.ipynb : pickle 파일 내용 확인을 위한 코드
│   ├── predict.ipynb : 모델을 이용한 예측 Application 코드
│   ├── best_results : 모델 저장
│   │   ├── 20230807_165319.pkl
│   │   └── model_20230807_165319.pt
│   └── data : 데이터 전처리 코드 및 전처리한 결과
│       ├── merged_data_KW.xlsx
│       ├── merged_data_MWH.xlsx
│       ├── preprocessing.ipynb : 데이터셋 전처리 코드
│       ├── test_for_0901.xlsx
│       ├── test_for_0901_withseq.xlsx
│       └── dataset_example
│           └── 2022-08-31.xlsx
```

* **모델 학습은 jupyter notebook을 사용하였다. 아래부터는 전반적인 코드 실행 과정에 대한 설명으로, 보다 자세한 사항은 각 코드의 주석을 참고하기를 바란다.**

## 코드 실행 과정

### 데이터 전처리

1. 대회에서 제공한 데이터 기간(2022.07.01~2022.08.31)에 맞는 날씨 정보(날짜, 온도, 습도)를 기상청 홈페이지로부터 다운로드한다. 엑셀파일로 다운로드하고, 해당 데이터는 [https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36)로부터 얻을 수 있다.
2. `data/preprocessing.ipynb` 파일을 연다.
3. 2번째 셀까지 실행하되 `folder` 변수와 `save` 변수에 원하는 경로를 설정한다.
4. 해당 경로의 파일을 열어 한 날짜에 대한 유효전력 데이터가 잘 추출되었는지 확인한다.
5. 다음 셀을 실행하여 22년 7월 1일부터 8월 31일까지 각각의 날짜에 대한 유효전력 데이터만을 추출한다. 이 결과 62개의 엑셀 파일이 생성된다.
6. 마지막 셀까지 실행하되 `power_data_folder`, `weather_data_file`은 필요한 데이터들이 존재하는 경로로 알맞게 설정해주어야 하며, `save_file`에는 원하는 경로를 설정한다.

### 모델 학습 과정

1. `main.ipynb` 파일을 연다.
2. 3번째 셀의 Hyperparameter 값을 원하는대로 설정한다. 각 파라미터에 대한 설명은 주석을 참고한다. thread를 사용할 경우 `num_workers=8`로 설정되어 있다. 학습시킨 모델을 더 학습시키고 싶다면, `pretrained_model_path`에 해당 모델의 경로를 지정하면 된다.
3. 모델 학습 결과는 `results_folder = "/home/kimyirum/EMS/ict-2023-ems/load/results/"`로 지정되어 있기 때문에 해당 경로는 원하는 위치로 설정해준다.
4. 전체 셀을 실행하여 모델을 학습시킨다. 평가 결과는 MAE, MSE, RMSE로 측정하였다.
5. 마지막 셀은 하이퍼파라미터 최적화하는 코드로, optuna 라이브러리를 활용하였다. 해당 레포지토리에 업로드한 모델은 이미 최적화 과정을 거쳐 얻은 하이퍼 파라미터이지만, 실험을 더 진행하고 싶다면 `do_optuna = True`로 설정하고 `lr, hidden_dim, n_layers, batch_size` 변수에 원하는 boundary를 설정해주면 된다.

### 모델의 예측 Application

1. `predict.ipynb` 파일을 연다.
2. 2번째 셀에서 학습을 진행한 모델의 경로를 설정해준다. `path` 변수에는 모델이 저장된 폴더를, `name` 변수에는 원하는 pre-trained 모델의 이름을 설정한다.
