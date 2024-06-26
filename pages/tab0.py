from utils.app.io import *
from utils.data.dataloader import *
from utils.data.preprocessing import *
from missingno.missingno import matrix


def show_overview(metas, datas, data_id):
    pp = Preprocessor(metas)
    pp.fit = True
    
    meta, data = metas[data_id], datas[data_id]
    data = pp.set_dtypes(data, meta)

    st.subheader("1. Overview")
    st.write("#### 1.1 Data")

    with st_stdout("code"):
        data.info()
    st.dataframe(data.head())

    p = matrix(data, figsize=PARAMS.figsize)
    p.set_title("Missing data (black: not null row, white: null row)")
    st.pyplot(p.figure)

    st.write("#### 1.2 Meta data")
    with st_stdout("code"):
        meta.info()
    st.dataframe(meta.head())
    p = matrix(meta, label_rotation=0, figsize=PARAMS.figsize)
    p.set_title("Missing data (black: not null row, white: null row)")
    st.pyplot(p.figure)


def show_feature_distribution(metas, datas, data_id):
    st.subheader("")
    st.subheader("2. Features Distribution")
    st.write("#### 2.1 Numerical Features")
    show_numerical_features(metas, datas, data_id)

    st.write("#### 2.2 Categorical Features")
    show_categorical_features(metas, datas, data_id)
    
    
def show_target_distribution(metas, datas, data_id, d):
    st.subheader("")
    st.subheader("3. Sample Target EDA")
    show_target_eda(metas, datas, data_id, d)

