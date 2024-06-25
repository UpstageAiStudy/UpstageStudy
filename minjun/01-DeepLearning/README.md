# Week1

## Missions

Deep Learning의 기본에 대해 공부하고 관련된 미션들을 풀어보자!

### Mission 1(난이도 하)

1. Pytorch Template [https://github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template) 을 이용하여 MNIST를 분류하는 MLP 모형 만들기

2. Pytorch 공식 튜토리얼 문서의 컴퓨터 비전 전이 학습 [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)을 이해하고 각 줄에 대한 주석 달기

### Mission 2(난이도 중)

1. Convolutional Neural Networks를 직접 구성하여 99% 이상의 성능을 내는 MNIST 분류기 만들기

2. Recurrent Neural Networks (RNN or LSTM or GRU)를 직접 구성하여 98% 이상의 성능을 내는 MNIST 분류기 만들기

3. Albumentation [https://albumentations.ai/](https://albumentations.ai/) 라이브러리를 이용하여 MNIST 데이터를 증강하여 99.5% 이상의 성능을 내는 MNIST 분류기 만들기

### Mission 3(난이도 상)

1. Convolution과 Activation 레이어만을 활용하여 MNIST 분류기 만들기

   - Flatten 연산 및 Fully Connected 레이어 없이 CNN을 만들기 위해서는 Global Average Pooling을 이용해 (b, 1, 1, dim)의 형태로 만든다.
   - 이후 1 x 1 conv를 사용해서 (b, 1, 1, num_classes) 형태로 바꿔준다.

2. Semi-supervised learning을 이용한 MNIST 분류기 만들기
   - 참고1 : [https://blog.est.ai/2020/11/ssl/](https://blog.est.ai/2020/11/ssl/)
   - 참고2 : [https://github.com/rubicco/mnist-semi-supervised](https://github.com/rubicco/mnist-semi-supervised)

## 내가 만든 퀴즈

- 문제 1번 : 역전파 알고리즘의 동작원리를 간략하게 설명하시오.
  Cost 함수에 대한 출력층의 미분(Gradient)를 계산하고, 이를 하류로 흘려보내면서 시작한다.

  $$
  \frac{\partial J}{\partial A^L}
  $$

  $$
  \frac{\partial J}{\partial Z^L}=\frac{\partial J}{\partial A^L}  \odot \frac{\partial A^L}{\partial Z^L}
  $$

  $$
  \frac{\partial J}{\partial W^L} = \frac{\partial J}{\partial Z^L} \cdot \frac{\partial Z^L}{\partial W^L}, \ \ \
  \frac{\partial J}{\partial b^L} = \frac{\partial J}{\partial Z^L} \cdot \frac{\partial Z^L}{\partial b^L},
  \ \ \
  \frac{\partial J}{\partial A^{L-1}} = \frac{\partial J}{\partial Z^L} \cdot \frac{\partial Z^L}{\partial A^{L-1}}
  $$

- 문제 2번 : 역전파 단계에서 기울기(gradient)가 소실되는 원인 두 가지를 설명하시오.

$$
\frac{\partial J}{\partial W^l} =
\frac{\partial J}{\partial Z^l} \cdot
\frac{\partial Z^l}{\partial W^l} =
\frac{\partial J}{\partial A^l}
\cdot
\frac{\partial A^l}{\partial Z^l}
\cdot
{X^l}^T
$$

- 문제 3번 : Adam 함수는 momentum과 rmsprop을 모두 종합한 것이라고 배웠는데, momentum과 rmsprop이 각각 어떤 기능인지 설명하시오.
  - momentum : 이전 단계에서 발생했던 기울기를 누적한 것과 현재 단계의 기울기를 적절히 조합한 방식으로 가속도가 적용되어 local minima나 saddle point를 벗어나는데 유효함.
  - rmsprop : 모델이 가진 학습 파라미터들은 서로 변화률(업데이트)이 재각각이다. rmsprop은 변화률이 큰 가중치는 적게 바뀌게 함으로써 파라미터들간 변화률을 비슷하게 만들어 준다.
