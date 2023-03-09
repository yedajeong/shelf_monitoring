data: 진열대 크롭 + object bbox(.xml)

data_cluster: 0~13으로 clustered object bbox만 (filename_c{cluster}.jpg)

data_obj: object bbox만 크롭 (filename_c{cluster}.jpg)

results_(num): clustering 결과 저장 (0~13, cluster내의 data 20개까지 clipping해서 plot)

# image_clustering.py
feature extraction - pca - kmeans

# cluster_view.py
image clustering 결과 visualization

# vgg16.py
vgg16 -> gs 데이터로 fine tuning

# file_handling.py
필요한 파일/폴더 생성, 이동, 저장, 복사 .. (temp)

# detector_cluster_local.py
백엔드 적용 전 로컬에서 테스트용
is_test==False인 부분 주석처리