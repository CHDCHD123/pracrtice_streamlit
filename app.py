import streamlit as st
import pandas as pd
import folium
import json
import numpy as np
import joblib
from streamlit_folium import st_folium
from shapely.geometry import shape, Point
from catboost import CatBoostRegressor

# st.set_page_config(layout='wide')
#-------------------main page-------------------
st.image("airbnb_logo.png", width=150)
st.markdown("""
### 💰 숙소 가격 예측 결과 확인
사이드 바에서 모든 변수 설정 완료 후 가격 예측을 진행해주세요

""")


#############################
# CSV 불러오기
categories_df = pd.read_csv("amenities.csv")

# 카테고리 리스트 추출
category_options = categories_df['amenities'].tolist()

# 뉴욕 지역이 표시된 맵을 띄우기 위해 데이터프레임 생성
df = pd.read_csv('2025_Airbnb_NYC_listings.csv', index_col=0)

new_df = df[['id','room_type', 'longitude', 'latitude', 'accommodates', 'minimum_nights','maximum_nights', 
            'bedrooms', 'bathrooms', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 
            'review_scores_rating', 'beds', 'instant_bookable', 'amenities', 'price']]

# 가격 데이터 범주형 -> 수치형 전환
new_df['price'] = new_df['price'].replace(r'[\$,]', '', regex=True).astype('float')

price_threshold = 472.5
new_df['type'] = new_df['price'].apply(lambda x:'high' if x>price_threshold else 'low')

geo_df = new_df.groupby(['neighbourhood_group_cleansed', 'type']).size()
geo_df = geo_df.reset_index()
geo_df.columns = ['neighbourhood_group_cleansed', 'type', 'count']
df_vs = geo_df.pivot(index='neighbourhood_group_cleansed', columns='type', values='count')
df_vs['vs'] = df_vs['high']-df_vs['low'] < 0
df_vs['vs'] = df_vs['vs'].astype(float)
df_vs = df_vs.reset_index()

map_center_lat = new_df['latitude'].mean()
map_center_lon = new_df['longitude'].mean()

geo_json = json.load(open('nyc.geojson', encoding="utf-8"))
m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=10)

folium.Choropleth(
    geo_data=geo_json,
    name='choropleth',
    data=df_vs,
    columns=['neighbourhood_group_cleansed', 'vs'],
    key_on='feature.properties.boroname',
    fill_color='BuGn',
    fill_opacity=0.7,
    line_opacity=0.5,
    line_color='#000000',
    line_weight=3,
).add_to(m)

#############################

# --- 1. 모델 및 칼럼 불러오기 ---

data = joblib.load("model_all2.pkl")


model = data["model"]
X_columns = data["X_columns"]
cat_features = data["cat_features"]
text_features = data["text_features"]
std_residual = data["std_residual"]  

# --- 2. 피처등 ---

with st.sidebar:
        st.header("Parameters")
        
        #room_type
        room_type = st.selectbox('룸 타입을 선택하세요',['Private room', 'Entire home/apt', 'Hotel room', 'Shared room'])

        #accommodates 투숙객 수
        accommodates = st.slider(
            label="인원수를 선택하세요",
            min_value=1,
            max_value=16,
            value=6,
            step=1,
        )
        # minimum_nights
        minimum_nights = st.slider(
            label="최소 숙박일 수를 선택하세요",
            min_value=1,
            max_value=730,
            value=300,
            step=1,
        )
        
        # maximum_nights
        maximum_nights = st.slider(
            label="최대 숙박일 수를 선택하세요",
            min_value=1,
            max_value=1125,
            value=500,
            step=1,
        )
        bedrooms = int(st.selectbox('Bedrooms', ['0','1','2','3','4','5','6']))
        bathrooms = float(st.selectbox('Bathrooms', ['0', '0.5', '1.0', '1.5', '2.0', '2.5']))
        beds = int(st.selectbox('Beds', ['0','1','2','3','4','5','6','7','8']))
        
        # review_scores_rating
        review_scores_rating = st.slider(
            label="기대 리뷰 평점을 선택하세요",
            min_value=1.00,
            max_value=5.00,
            value=3.00,
            step=0.01,
        )
        # 멀티셀렉트 추가
        amenities = st.multiselect(
            "편의시설을 선택하세요",
            options=category_options
        )
        
        # instant_bookable
        instant_bookable = st.radio('instant_bookable',options=['Yes', 'No'])
        instant_bookable = True if instant_bookable == 'Yes' else False

    

        # This code snippet is responsible for displaying a map using Folium in Streamlit and
        # capturing user interactions with the map. Here's a breakdown of what each part does:
        # 지도 출력
        map_data = st_folium(m, width=500, height=300)

        # 📁 거리 정보 데이터 불러오기
        proximity_df = pd.read_csv("mapbox.csv")  # 이 파일에 subway, bus, airport 거리 정보 있음

        merged_df = new_df.merge(proximity_df, how='left', on='id')

        # 📍 지도 클릭 정보
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lon = map_data["last_clicked"]["lng"]
            st.write(f"🧭 클릭한 위치의 위도/경도: `{lat:.6f}`, `{lon:.6f}`")

            clicked_point = Point(lon, lat)
            matched_region = None

            for feature in geo_json["features"]:
                polygon = shape(feature["geometry"])
                if polygon.contains(clicked_point):
                    matched_region = feature["properties"].get("boroname")
                    matched_region2 = feature["properties"].get("ntaname")
                    break

            if matched_region:
                st.write(f"🏘️ 클릭한 지역은 `{matched_region}의 {matched_region2}`입니다.")

                # ✅ 가장 가까운 숙소 한 개 찾기 (Euclidean 거리 기준)
                merged_df["distance_to_click"] = ((merged_df["latitude"] - lat) ** 2 + (merged_df["longitude"] - lon) ** 2)
                closest_row = merged_df.loc[merged_df["distance_to_click"].idxmin()]

                # ✅ 거리 정보 추출
                subway_dist = closest_row.get("walk_subway(m)", None)
                bus_dist = closest_row.get("walk_bus(m)", None)
                airport_drive = closest_row.get("car_airport(m)", None)

                st.markdown("### 🚏 교통 거리 정보")
                walk_speed_m_per_min = 80         # 도보 속도 (m/min)
                drive_speed_m_per_min = 666.67    # 운전 속도 (m/min)

                def format_minutes(mins):
                    mins = int(round(mins))  # 반올림 후 정수
                    if mins >= 60:
                        hours = mins // 60
                        minutes = mins % 60
                        return f"{hours}시간 {minutes}분"
                    else:
                        return f"{mins}분"

                # 🚇 지하철
                if pd.notna(subway_dist):
                    time_to_subway = subway_dist / walk_speed_m_per_min
                    st.markdown(
                        f"🚇 가장 가까운 **지하철역까지**:<br>"
                        f"{subway_dist:.0f} m (도보 약 {format_minutes(time_to_subway)})",
                        unsafe_allow_html=True)
                else:
                    st.write("🚇 지하철 거리 정보 없음")

                # 🚌 버스
                if pd.notna(bus_dist):
                    time_to_bus = bus_dist / walk_speed_m_per_min
                    st.markdown(
                        f"🚌 가장 가까운 **버스 정류장까지**:<br>"
                        f"{bus_dist:.0f} m (도보 약 {format_minutes(time_to_bus)})",
                        unsafe_allow_html=True
                    )
                else:
                    st.write("🚌 버스 정류장 거리 정보 없음")

                # ✈️ 공항
                if pd.notna(airport_drive):
                    drive_time = airport_drive / drive_speed_m_per_min
                    st.markdown(
                        f"✈️ **공항까지 운전 거리**:<br>"
                        f"{airport_drive:.0f} m (운전 약 {format_minutes(drive_time)})",
                        unsafe_allow_html=True
                    )
                else:
                    st.write("✈️ 공항 거리 정보 없음")
            else:
                st.write("❌ 클릭한 위치가 어떤 지역에도 속하지 않습니다.")

#----------------------슬라이드 마감 ------------------------------

#-------------------main page-------------------

if st.button("가격 예측 실행하기✔️"):
    # amenities 문자열 변환
    amenities_str = ', '.join(amenities)

    # 입력 데이터프레임 생성
    input_df = pd.DataFrame([{
        'room_type': room_type,
        'longitude': lon,
        'latitude': lat,
        'accommodates': accommodates,
        'minimum_nights': minimum_nights,
        'maximum_nights': maximum_nights,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'neighbourhood_cleansed': matched_region2,
        'neighbourhood_group_cleansed': matched_region,
        'review_scores_rating': review_scores_rating,
        'beds': beds,
        'instant_bookable': instant_bookable,
        'walk_subway(m)': subway_dist,
        'walk_bus(m)': bus_dist,
        'car_airport(m)': airport_drive, 
        'amenities': amenities_str
    }])

    # 컬럼 순서 맞추기
    input_df = input_df[X_columns]

    # 예측 수행
    log_price = model.predict(input_df)[0]
    pred_price = np.expm1(log_price)  # 로그 복원

    # 오차 범위 계산
    lower = np.maximum(0, pred_price - 0.5 * std_residual)
    upper = pred_price + 0.5 * std_residual

    # 결과 출력
    st.success(f"🏷️ 예측 숙소 가격: **${pred_price:,.2f}**")
    st.info(f"📏 예측 오차 범위: {lower:,.2f} ~ {upper:,.2f}")

    with st.expander("입력 확인"):
        st.dataframe(input_df)
    
st.markdown("""
--- 
##### 🧠 예측 모델 정보

""")


#-------------------main page + 상세 페이지 부분 -------------------

with st.expander("🔎 어떤 항목들이 예측에 사용되나요?"):
    st.markdown("""
    - 숙소 유형 (Entire home, Private room 등)
    - 위치 (위도/경도)
    - 숙박 가능 인원 수
    - 욕실, 침실 수
    - 최소 / 최대 숙박일
    - 즉시 예약 가능 여부
    - 리뷰 평점
    - 제공 어메니티 수
    """)

with st.expander("📊 모델 정보"):
    st.markdown("""
    - 알고리즘: CatBoost Regressor
    - 평가 지표: RMSE = 23.5, MAE = 18.1, R2 = 0.93
    - 로그 변환(y) + 오차범위 표시 기능 포함
    """)

st.markdown("""
---
##### 📌 프로젝트 정보

👨‍💻 프로젝트명 | 2025 내일배움캠프 심화 프로젝트  
👥 팀 멤버 | 김윤환, 김소영, 조현도, 임은지  
📁 데이터 출처 | Airbnb (크롤링 데이터 기반)  
📬 문의 메일 | `8조_세계정복김순대@gmail.com`
""")