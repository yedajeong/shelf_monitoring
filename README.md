# shelf_monitoring
: __무인매장 재고 관리를 위한 AI 결품감지 시스템__ <br/> <br/>

진열대 내의 상품이 판매되어 결품이 발생했을 때 이를 감지하고 실시간 결품 보충작업 지시를 내릴 수 있습니다. <br/>

- 결품감지 결과 예시



### 1. 결품감지 성능 개선
- 학습용 데이터셋 라벨링 (pascal voc)
- object detection 모델 학습 (retinanet 기반)
- 학습 데이터 추가 전 / 후


- 모델 파라미터 값 튜닝, 로직 수정
- 성능 개선 전 / 후
- 






### 2. 결품감지 기능 추가
- 다른 카테고리의 상품이 위치 시 이를 탐지, 결품이라 판단
- pre-trained foundation 모델인 vgg16을 전이학습
- 상위 6개의 layer(block4 conv layer 3개 + block5 conv layer 3개)만 unfreeze시켜 도메인 데이터를 지도학습시킨 fine tuning 모델
- 이를 feature extractor로써 사용
- 추출된 feature간의 cosine similarity를 기반으로 일정 threshold 이하의 유사도인 경우 다른 상품으로 판단, 결품이라고 파악함

- vgg16 fine tuning 학습용 데이터셋: 14종류 상품 6면을 촬영한 이미지 데이터셋
- 유사도 상위 20개의 상품 추출 결과 예시


- 이전 프레임 & 현재 프레임 다른 상품이 놓인 경우 bbox로 표시

