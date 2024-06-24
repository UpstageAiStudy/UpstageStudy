# UpstageStudy 11조

## Installation

    ## Pytorch
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

    ## Others
    pip install -r requirements.txt

## [24.06.18 ~ 24.06.24] 01.DeepLearning

- Upstage AI Lab에서 Deeplearning 파트 강의를 듣고 관련 미션들을 직접 해결. [01-DeepLearning](./01-DeepLearning)
- 수업 내용에서 중요하다고 생각된 부분을 퀴즈로 만들기.

  - 문제 1번 : 역전파 알고리즘의 동작원리를 간략하게 설명하시오.
    Cost 함수에 대한 출력층의 미분(Gradient)를 계산하고, 이를 하류로 흘려보내면서 시작한다.
    $$
    \frac{\partial J}{\partial A^L}
    \\
    \
    \\
    \frac{\partial J}{\partial Z^L}=\frac{\partial J}{\partial A^L}  \odot \frac{\partial A^L}{\partial Z^L}
    \\
    \
    \\
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
