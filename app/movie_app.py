import streamlit as st
import joblib, os
from dotenv import load_dotenv
from typing import List, Dict, Any
from tmdbv3api import Movie, TMDb
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# 기본 정보 세팅
load_dotenv()
API_KEY = os.getenv("API_KEY")

movie = Movie()
tmdb = TMDb()
tmdb.api_key = API_KEY

# pkl 객체 load
movie_master = pd.read_parquet('../dataset/tmdb_movies.parquet')[['id', 'title']]
contents_cos_sim = joblib.load('../models/contents_cos_sim.pkl')  # 줄거리 기반 유사도
info_cos_sim = joblib.load('../models/info_cos_sim.pkl')  # 등장인물 기반 유사도

lang_dict = {
    "English": "en-US",
    "한국어": "ko-KR"
}


def get_movie_info(movie_id: int) -> Dict[str, Any]:
    """
    영화 상세 정보 가져오기
    """
    try:
        movie_details = movie.details(movie_id)
        
        # 점수 (TMDB 평점)
        score = movie_details.get('vote_average', 0)
        
        # 개봉일
        release_date = movie_details.get('release_date', 'Unknown')
        if release_date and release_date != 'Unknown':
            release_date = release_date[:4]  # 연도만 추출
        else:
            release_date = 'N/A'
        
        # 장르
        genres = movie_details.get('genres', [])
        genre_names = [genre.get('name', '') for genre in genres if genre.get('name')]
        
        # 포스터 경로
        poster_path = movie_details.get('poster_path', '')
        if poster_path:
            poster_url = "https://image.tmdb.org/t/p/w500" + poster_path
        else:
            poster_url = "no_image.jpg"
        
        return {
            'title': movie_details.get('title', 'Unknown'),
            'score': score,
            'release_date': release_date,
            'genres': genre_names,
            'poster_url': poster_url
        }
    except Exception as e:
        st.warning(f"영화 정보를 가져오는 중 오류가 발생했습니다: {e}")
        return {
            'title': 'Unknown',
            'score': 0,
            'release_date': 'N/A',
            'genres': [],
            'poster_url': "no_image.jpg"
        }


def get_recommendation(
    movie_name: str,
    cos_similarity: np.ndarray,
    num_recommendations: int = 10
) -> List[Dict[str, Any]]:
    """
    영화 추천 함수 - 상세 정보 포함
    """
    movie_infos = []
    
    try:
        idx = movie_master[movie_master['title'] == movie_name].index[0]
    except IndexError:
        st.error(f"영화 '{movie_name}'를 찾을 수 없습니다.")
        return []

    sim_scores = list(enumerate(cos_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    movie_idx = [i[0] for i in sim_scores if i[0] != idx][:num_recommendations]

    for idx in movie_idx:
        movie_id = movie_master['id'].iloc[idx]
        movie_info = get_movie_info(movie_id)
        movie_infos.append(movie_info)
    
    return movie_infos


def display_movie_grid(
    movie_infos: List[Dict[str, Any]], 
    num_cols: int = 6
):
    """영화 그리드 표시 함수 - 상세 정보 포함"""
    if not movie_infos:
        st.warning("추천할 영화가 없습니다.")
        return
    
    # 6의 배수로 맞추기 위해 빈 공간 추가
    total_items = len(movie_infos)
    items_per_row = 6
    rows_needed = (total_items + items_per_row - 1) // items_per_row  # 올림 계산
    total_slots = rows_needed * items_per_row
    
    # 빈 공간을 채우기 위해 None 추가
    padded_movies = movie_infos + [None] * (total_slots - total_items)
    
    for i in range(0, total_slots, items_per_row):
        cols = st.columns(items_per_row)
        for j, col in enumerate(cols):
            if i + j < total_slots and padded_movies[i + j] is not None:
                movie_info = padded_movies[i + j]
                with col:
                    # 포스터 이미지
                    st.image(movie_info['poster_url'])
                    
                    # 영화 제목
                    st.text(movie_info['title'])
                    
                    # 점수, 개봉일, 장르 정보
                    st.markdown(
                        f"""
                        <div style='text-align: center; font-size: 12px; color: #666;'>
                            ⭐ {movie_info['score']:.1f} |  {movie_info['release_date']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # 장르 정보
                    genres_text = ', '.join(movie_info['genres'][:2]) if movie_info['genres'] else 'No genre info'
                    st.markdown(
                        f"""
                        <div style='text-align: center; font-size: 11px; color: #888; margin-top: 2px;'>
                            {genres_text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                # 빈 공간
                with col:
                    st.empty()


# 페이지 설정
st.set_page_config(
    page_title="영화 추천 시스템",
    page_icon="🎬",
    layout="wide"
)

# 메인 타이틀
st.title("🎬 영화 추천 시스템")
st.markdown("---")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 언어 선택
    selected_lang = st.selectbox(
        "언어 선택",
        options=list(lang_dict.keys()),
        index=0
    )
    if selected_lang:
        tmdb.language = lang_dict[selected_lang]
    
    # 추천 개수 선택
    num_recommendations = st.slider(
        "추천 영화 개수",
        min_value=5,
        max_value=36,
        value=12
    )

# 메인 컨텐츠
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎬 영화 선택")
    movie_list = movie_master['title'].tolist()
    selected_movie = st.selectbox(
        "좋아하는 영화를 선택하세요",
        options=movie_list,
    )

with col2:
    st.subheader("ℹ️ 선택된 영화")
    if selected_movie:
        st.info(f"**{selected_movie}**")
        st.caption("이 영화와 유사한 영화들을 추천해드립니다!")

st.markdown("---")

# 탭으로 추천 방식 구분
tab1, tab2 = st.tabs(["📖 줄거리 기반 추천", "👥 등장인물 기반 추천"])

with tab1:
    if st.button("📖 줄거리 기반 추천 실행", use_container_width=True):
        st.subheader("📖 줄거리 기반 추천 결과")
        with st.spinner("추천 영화를 찾는 중..."):
            if selected_movie:
                movie_infos = get_recommendation(
                    selected_movie, 
                    contents_cos_sim, 
                    num_recommendations
                )
                display_movie_grid(movie_infos)

with tab2:
    if st.button("👥 등장인물 기반 추천 실행", use_container_width=True):
        st.subheader("👥 등장인물 기반 추천 결과")
        with st.spinner("추천 영화를 찾는 중..."):
            if selected_movie:
                movie_infos = get_recommendation(
                    selected_movie, 
                    info_cos_sim, 
                    num_recommendations
                )
                display_movie_grid(movie_infos)

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎬 영화 추천 시스템 | Powered by TMDB API</p>
    </div>
    """,
    unsafe_allow_html=True
)