# Hand2ID
personal hand writting as ID, with changing hyperparmeters 
This work is just personal project to make a programm with ideas in my head. 해당 작업은 단순히 개인적인 아이디어를 프로그램화 해보는 개인 프로젝트입니다.

Personal hand-writting indicating that can be used as ID. By using two or three pictures of the person's handwriting(currently only with number 0 to 10). 개인의 손글씨를 ID처럼 사용하는 작업으로 현재는 숫자 0부터 9까지를 활용. 개인의 손글씨 이미지 2~3장만을 활용하여 ID화하는 작업 Summary Machanizim follows: 전체적인 방법은 다음과 같습니다.

1.import 0 to 9 handwritting image and transfrom into corrdinate data. 
  0에서 9까지의 손글씨 이미지를 불러와 좌표화 시킴.
2.with corrdinate data, random sample and spin them to boost the number of sample size. 
    좌표에서 랜덤 샘플링을 하여서 샘플의 수를 증폭시킴
4. do simple CNN for 0 to 9 with changing hyper-parameters
    hyper-parameter을 변경하여 단순한 CNN을 지속적으로 돌림
5. find the best hyperparameter
   해당 CNN 모델의 스코어가 가장 높은 hyperparameter을 저장
6. best hyperparmeter and the simple CNN creates the ID for the person
   가장 높은 hyperparamter와 CNN 모델이 ID.
