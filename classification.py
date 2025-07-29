import streamlit as st
import pickle

# Load the trained model
with open("./mymodel/classification.pkl", "rb") as file:
    model = pickle.load(file)

# Configure Streamlit app
st.set_page_config(page_title='5계 분류 예측', layout='wide')
st.title('생물 5계 분류 예측 모델')

# Additional information
with st.expander("5계 분류란?"):
    st.write("""
    지구상에 존재하는 생물을 원핵생물계, 원생생물계, 식물계, 동물계, 균계라는 5계로 구분됩니다. 
    이 모델은 특정한 생물이 가진 특징들을 활용하여 해당 생물이 5계 중 어떤 계에 속하는 지 예측하는
    모델입니다.
    """)

# Input form
with st.form("생물의 특징 입력하기"):
    st.subheader("특징 입력")
    nucleus = st.number_input("핵의 유무 (0: 없음, 1: 있음):", min_value=0, max_value=1, step=1)
    cell_wall = st.number_input("세포벽의 유무 (0: 없음, 1: 있음):", min_value=0, max_value=1, step=1)
    photosynthesis = st.number_input("광합성 여부 (0: 광합성을 안함, 1: 광합성을 함):", min_value=0, max_value=1, step=1)
    motility = st.number_input("운동성 여부 (0: 없음, 1: 있음):", min_value=0, max_value=1, step=1)
    cell_count = st.number_input("세포의 수 (1: 1개(단세포), 2: 여러개(다세포)):", min_value=1, max_value=2, step=1)
    stem_leaf_root = st.number_input("줄기/잎/뿌리의 유무 (0: 발달하지 않음, 1: 발달함):", min_value=0, max_value=1, step=1)

    submitted = st.form_submit_button("예측 결과")

    if submitted:
        # Make prediction
        features = [nucleus, cell_wall, photosynthesis, motility, cell_count, stem_leaf_root]
        prediction = model.predict([features])[0]
        st.markdown(f"이 생물은 <span style='color:red;'>{prediction}</span>에 속할 것으로 예상됩니다.", unsafe_allow_html=True)

with st.expander("특징과 5계의 관련성"):
        plusexp = '''
생물의 특징들과 5계의 상관관계를 보여주는 이미지
        '''
        
        st.markdown(plusexp)
        st.image('./static/images/classficaiton_correlation.png')
