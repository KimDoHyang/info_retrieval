# RESULT

> **Parameter Configuratoins**
>
> 1000000 lines of text8.txt file
>
> window size = 5
>
> negative samples 10
>
> subsampling threshold 0.001
>
> hidden layer dimension size 100
>
> epoch 1
>
> fnv-1a hash space 2^32



결과는 loss가 크게 나와 비교적 부정확하게 나타났으나, 이는 학습 데이터 양이 매우 적기 때문으로 예상된다.

실제로 1000000 라인이 아닌 전체 텍스트 데이터를 학습시킬 경우에는 loss가 이전보다 크게 떨어지는 것이 확인되었다.

학습에 소요되는 시간이 많아 전체 데이터 및 epoch의 수를 늘리지는 못해 더 향상된 결과를 제출하지 못한 점이 아쉽다.

