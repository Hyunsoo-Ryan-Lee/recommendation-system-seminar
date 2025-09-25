# 🎬 추천 시스템 A to Z: 개념부터 웹앱 구현까지

## 📁 프로젝트 구조

```
recommendation-system-seminar/
├── dataset/                         # 데이터셋 폴더
│   ├── products.csv                 # 상품 데이터
│   ├── users.csv                    # 사용자 데이터
│   ├── ratings.csv                  # 평점 데이터
│   ├── tmdb_movies.parquet          # TMDB 영화 데이터
│   └── tmdb_credits.parquet         # TMDB 출연진 데이터
│
├── models/                          # 학습된 모델 저장 폴더
│
├── 01_recommendation.ipynb          # 기본 추천 시스템 모델 실습
├── 02_tmdb_줄거리기반추천.ipynb       # TMDB 줄거리 기반 추천 구현
├── 03_tmdb_인물기반추천.ipynb         # TMDB 인물 기반 추천 구현
├── movie_app.py                     # Streamlit 웹 애플리케이션
├── requirements.txt                 # 필요한 라이브러리 목록
└── README.md                        
```
## 📊 데이터셋 설명

### 1) 상품 추천 데이터셋
- **products.csv**: 상품 정보 (ID, 이름, 카테고리, 브랜드, 설명, 태그)
- **users.csv**: 사용자 정보 (ID, 나이, 성별, 직업, 우편번호)
- **ratings.csv**: 평점 데이터 (사용자 ID, 상품 ID, 평점, 타임스탬프)

### 2) 영화 추천 데이터셋 (TMDB)
- **tmdb_movies.parquet**: 영화 기본 정보
  - `id`: 영화 고유 ID
  - `title`: 영화 제목
  - `overview`: 영화 줄거리
  - `genres`: 영화 장르
  - `vote_average`: 평균 평점
  - `popularity`: 인기도

- **tmdb_credits.parquet**: 출연진 정보
  - `movie_id`: 영화 ID
  - `cast`: 출연진 정보
  - `crew`: 제작진 정보

## 📋 웹앱 개발 프로젝트 개요

### 추천 알고리즘
1. **내용 기반 추천 (Content-Based Filtering)**
   - TF-IDF 벡터화를 통한 텍스트 특성 추출
   - 코사인 유사도 기반 영화 유사도 계산
   - 영화 줄거리와 장르 정보 활용

2. **인물 기반 추천 (Cast & Crew Based)**
   - 감독, 배우 등 인물 정보를 활용한 추천
   - TF-IDF 벡터화를 통한 인물 정보 특성 추출

## 🚀 설치 및 실행

### 1. 기술 스택

- **Python 3.8+**
- **데이터 처리**: pandas, numpy
- **머신러닝**: scikit-learn
- **웹 애플리케이션**: Streamlit
- **API**: TMDB API
- **개발 환경**: Jupyter Notebook

### 2. 환경 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 필요한 라이브러리 설치
pip install -r requirements.txt
```

### 3. 웹 애플리케이션 실행

```bash
# Streamlit 웹 애플리케이션 실행
streamlit run movie_app.py
```


## 🌐 웹 애플리케이션 사용법

1. **영화 선택**: 드롭다운에서 원하는 영화를 선택
2. **언어 설정**: 한국어 또는 영어 선택
3. **추천 개수**: 5~36개 중 추천할 영화 개수 선택
4. **추천 실행**: 
   - 줄거리 기반 추천 → 영화 줄거리와 장르를 기반으로 추천
   - 등장인물 기반 추천 → 감독, 배우 정보를 기반으로 추천