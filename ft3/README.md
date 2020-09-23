# Text Classification Assignment
2011210078 장두호

## Execution & Requirement
실행을 위해서는 torch & torchvision의 설치가 요구됩니다.
구현에 사용된 Python 버전은 3.7.2입니다.
```
python main.py {n-gram} {classes-path} {train-path} {test-path}
# n-gram: length of word n-gram
# classes-path: classes info file path
# train-path: file path of training data (csv)
# test-path: file path of test data (csv)
# ex) python main.py 2 classes.txt train.csv test.csv 
```

## Implementations
### classes.py
정답 class 정보를 로드합니다.

### data.py
csv 파일을 읽고 class, title, description 정보를 로드합니다.
또한, title 및 description을 word n-gram으로 변환합니다. 

### hash.py
주어진 word n-gram들을 FNV-1A 해싱하여 index로 치환합니다.

### train.py
주어진 data로부터 input / output sequence 를 추출하고,
해당 training data를 바탕으로 text classification model을 학습합니다.

### test.py
학습된 모델로 test data에 대한 classification을 수행하고 결과를 파일로 저장합니다.

### main.py
execution option(arguments)을 파싱하고 해당 조건에 맞게 학습 및 테스트를 실행합니다.
