# GIST ict-2023 EMS-Track: Team MYsolar

GIST 창의융합경진대회 2023 EMS-Track, MYsolar 팀 GitHub Repo

## Set-ups

아나콘다(Anaconda) 설치 링크: [Anaconda 공식문서](https://conda.io/projects/conda/en/latest/user-guide/install/download.html)

> conda version: `23.1.0`

설치 후, 가상환경 생성 및 패키치 다운로드

```bash
conda create -n ENVIRONMENT_NAME python==3.9
pip install -r requirements.txt
```

## Files & Program execution

### 예측 알고리즘

LSTM 모델을 사용하여, 다음날 24시간의 시간당 발전량/전력사용량 예측을 위해 지난 일주일간의 시간대별 데이터로 학습한다. 손실을 줄이기 위해, 비교적 건물 수가 14개로 적었던 태양광 발전량 데이터는 건물별로 모델을 생성하여 학습하였으며, 건물 수가 많았던(56) 부하 데이터의 경우 컴퓨팅 자원의 한계와 실행시간의 비효율성을 고려하여 건물을 하나의 feature로 설정한다.

- sequence length: 24*7 (지난 일주일의 시간당 데이터)
- label length: 24 (다음날 시간당 예측값)
- hyper-parameters:
  - 태양광 발전량 예측모델:
    ```python
    config ={
    	"num_layers":7,
    	"hidden_size":128,
    	"batch_size":128,
    	"num_epochs":200
    }
    ```
  - 부하 예측 모델
    ```python
    config ={
    	"num_layers": ,
    	"hidden_size": ,
    	"batch_size": ,
    	"num_epochs": 
    }
    ```

학습 및 검증 후, `.pt` 확장자로 모델의 state를 체크포인트에 저장하였다.

저장된 체크포인트와 미리 저장한 검증 데이터셋을 활용, `predict.py`를 실행하여 **2022.08.31**의 시간당 전력사용량 및 태양광 발전량을 예측한다.

```bash
python predict.py --mode pv
python predict.py --mode load
```

### 스케줄링 알고리즘

파이썬 오픈소스 라이브러리 중,  `pulp`를 활용하여 선형계획법을 구성한다.

이를 정의하기 위하여 사용한 변수 및 함수는 아래와 같다.

1. 의사결정 변수(Decision Variables)

- `energy_bought`: ESS grid로부터 시간당 공급받는(import) 에너지
- `energy_sold`: ESS grid에 시간당 공급하는 에너지
- `energy_charged`: ESS grid에 시간당 충전되는 에너지
- `energy_discharged`: ESS grid에서 시간당 방전되는 에너지
- `battery_level`: 배터리의 상태, 시간대별 시작과 끝을 고려하여 총 25개의 인덱스를 가지도록 변수를 생성함 (0시 0분, 0시 59분 등)

2. 목적 함수(Objective Function)

- 전기요금을 최소화함 (`energy_bought`로 인한 요금 최소화)
- ESS에 저장하는 에너지로부터 이익을 최대화함

3. 제약조건(Constraints)

- 배터리 충전/방전 및 배터리 효율을 고려한 상태(level) 제약
- 에너지 균형

매개변수(Parameter)는 배터리 충·방전 효율, 전기요금, 배터리 최대용량, 그리고 예측모델이 산출한 부하 및 태양광 발전량 예측값 등이 있다.

아래 명령어로 `schedule.py` 파일을 실행한다.

```python
python schedule.py --mode eval
```
