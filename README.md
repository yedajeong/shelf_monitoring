# shelf_monitoring
🛒 __무인매장 재고 관리를 위한 AI 결품감지 시스템__ <br/> <br/>

![Python](https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED.svg?&style=for-the-badge&logo=Docker&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8.svg?&style=for-the-badge&logo=OpenCV&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243.svg?&style=for-the-badge&logo=NumPy&logoColor=white)
![visualstudiocode](https://img.shields.io/badge/visualstudiocode-007ACC.svg?&style=for-the-badge&logo=visualstudiocode&logoColor=white)
![MQTT](https://img.shields.io/badge/MQTT-660066.svg?&style=for-the-badge&logo=MQTT&logoColor=white)

---

### Summary
진열대 내의 상품이 판매되어 결품이 발생했을 때 이를 감지하고 실시간 결품 보충작업 지시를 내릴 수 있습니다. <br/>

- 결품감지 결과 예시 <br/>
![ex1](https://user-images.githubusercontent.com/49023751/223939840-c398b214-3ffd-4334-b740-881c7da2128d.png)
![ex2](https://user-images.githubusercontent.com/49023751/223939938-6a81c2ee-b931-4ed9-9429-2cb92fc28c12.png)
![ex3](https://user-images.githubusercontent.com/49023751/223939949-618b77f4-9613-43d2-9402-dbd6e9751908.png)

<br/>

---
### 1. 결품감지 성능 개선
- 학습용 데이터셋 라벨링 (pascal voc)
- object detection 모델 학습 (retinanet 기반)

<br/>

- 객체 탐지 학습 데이터 추가 전 / 후
<img width="452" alt="1-before" src="https://user-images.githubusercontent.com/49023751/223939988-f4317b57-3191-4067-afbc-83e7fe2a5be0.png">

<img width="452" alt="1-after" src="https://user-images.githubusercontent.com/49023751/223940001-8b2104ed-1c5f-4d9e-9a0b-f40b1d94a807.png">

<br/><br/>

- 성능 개선 - 모델 파라미터 값 튜닝, 로직 수정 전(오탐) / 후

<a href="https://github.com/anuraghazra/github-readme-stats">
  <img align="center" width="236" alt="1-before(3)" src="https://user-images.githubusercontent.com/49023751/223940803-65ea2f3f-bcf8-4812-aa71-4f1197992cf8.png" />
</a>
<a href="https://github.com/anuraghazra/convoychat">
  <img align="center" width="236" alt="1-after(3)" src="https://user-images.githubusercontent.com/49023751/223940866-b1351f55-37c5-43e2-bc35-db7c8f6c4293.png" />
</a>

<br/><br/>

<a href="https://github.com/anuraghazra/github-readme-stats">
  <img align="center" width="262" alt="1-before(4)" src="https://user-images.githubusercontent.com/49023751/223940920-24df8d61-98a7-4156-b4e9-4c2e9a320cbc.png" />
</a>
<a href="https://github.com/anuraghazra/convoychat">
  <img align="center" width="262" alt="1-after(4)" src="https://user-images.githubusercontent.com/49023751/223940978-ddf49e6b-ddab-43d8-8628-fd640db8671d.png" />
</a>

<br/><br/>

<a href="https://github.com/anuraghazra/github-readme-stats">
  <img align="center" width="295" alt="1-before(5)" src="https://user-images.githubusercontent.com/49023751/223941002-21a599f7-a168-4e4f-8480-ca995c94229f.png" />
</a>
<a href="https://github.com/anuraghazra/convoychat">
  <img align="center" width="295" alt="1-after(5)" src="https://user-images.githubusercontent.com/49023751/223941007-fccb73b5-8c27-4450-9afc-576e880db877.png" />
</a>
  
<br/><br/>

---
### 2. 결품감지 기능 추가
- 다른 카테고리의 상품이 위치 시 이를 탐지, 결품이라 판단
- pre-trained foundation 모델인 vgg16을 전이학습
- 상위 6개의 layer(block4 conv layer 3개 + block5 conv layer 3개)만 unfreeze시켜 도메인 데이터를 지도학습시킨 fine tuning 모델
- 이를 feature extractor로써 사용
- 추출된 feature간의 cosine similarity를 기반으로 일정 threshold 이하의 유사도인 경우 다른 상품으로 판단, 결품이라고 파악함
- vgg16 fine tuning 학습용 데이터셋: 14종류 상품 6면을 촬영한 이미지 데이터셋

<br/>

- 유사도 상위 20개의 상품 추출 결과 예시

<img width="300" alt="2-clstr4" src="https://user-images.githubusercontent.com/49023751/223941081-444f61c7-6a14-4268-b55a-2f8c812ab41b.jpg">

<img width="300" alt="2-clstr3" src="https://user-images.githubusercontent.com/49023751/223941084-bd5ac981-07b1-4695-b7eb-9441a06e6849.jpg">

<img width="300" alt="2-clstr2" src="https://user-images.githubusercontent.com/49023751/223941088-030fbc33-d13c-4883-9faf-99ddc5e53587.jpg">

<img width="300" alt="2-clstr1" src="https://user-images.githubusercontent.com/49023751/223941091-89857ea8-0402-46f3-8613-07bd1d131705.jpg">


<br/>

- 이전 프레임 & 현재 프레임 다른 상품이 놓인 경우 bbox로 표시 <br/>
<a href="https://github.com/anuraghazra/github-readme-stats">
  <img align="center" width="300" alt="result_before" src="https://user-images.githubusercontent.com/49023751/227785292-9b8c37ef-a27b-4870-a350-a9181a6939b6.png" />
</a>
<a href="https://github.com/anuraghazra/convoychat">
  <img align="center" width="300" alt="result_after" src="https://user-images.githubusercontent.com/49023751/227785302-cafe638b-926a-4046-8938-9514a50c3608.png" />
</a>
