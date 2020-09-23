# FastText Assignment
2011210078 장두호

## Execution & Requirement
실행을 위해서는 torch & torchvision의 설치가 요구됩니다.
구현에 사용된 Python 버전은 3.7.2입니다.
```
python main.py {ns} {gram_min} {gram_max} {part}
# ns: negative sample 수
# gram_min: 사용할 최소 subword 길이
# gram_max: 사용할 최대 subword 길이
# part: "part" 이면 부분 학습, "full" 이면 전체 학습
# ex) python main.py 20 3 6 full
```

## Result Screenshots
결과 스크린샷은 screenshots 디렉토리에 저장되어 있습니다.
negative sample 수는 20개이며, 3 ~ 6 길이의 n-gram을 사용하여 실험을 진행했습니다.
### Result 1
subsampling threshold = 0.001, hash bound K = 2.1 * 10^9

### Result 2
subsampling threshold = 0.001, hash bound K = 2 * 10^6

### Result 3
subsampling threshold = 0.0001, hash bound K = 2 * 10^6

## Implementations
### subword.py
word로부터 subword를 추출하는 로직이 구현되어 있습니다.

### corpus.py
주어진 corpus data path로부터 corpus 객체를 로드합니다.
word 및 subword를 추출하고, indexing & hashing을 통해 index number 로 치환합니다.
또한 training data를 생성하며, 이 때 subsampling이 적용됩니다.

### sampler.py
negative sampling 로직이 구현되어 있습니다.
가중치 리스트를 받아, binary search를 통해 해당 가중치에 비례하는 확률로 negative sample을 선택합니다.
이는 추후 fasttext 학습 시에 frequency^(3/4) 의 가중치 리스트로 사용됩니다.

### train.py
주어진 hyper parameter 로 FastText 모델을 학습합니다. 
Skip-gram & Negative Sampling 및 training 로직이 구현되어 있습니다.

### experiment.py
주어진 테스트 단어들에 대해 가장 유사한 5개 단어를 찾는 task 가 구현되어 있습니다.
subword embedding matrix 를 통해 word embedding 을 생성하고,
cosine similarity 를 기준으로 가장 유사한 5개 단어를 찾습니다.

### main.py
학습 및 실험을 실행하는 main script 입니다.
argument parsing, corpus loading, training, experiment를 순서대로 실행합니다.
