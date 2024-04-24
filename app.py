# original code https://github.com/alchemine/diabetes-prediction/blob/main/diabetes_prediction/app.py
# modified by Kyulim-Kim

from pages import *
from utils.data.preprocessing import *
import pickle


st.set_page_config(layout="wide", initial_sidebar_state='collapsed')

st.title('NHIS-2018 Data Analysis & Prediction')
tab_titles = ['Explanatory Data Analysis', 'Feature Engineering', 'Modeling']
tabs = st.tabs(tab_titles)


with tabs[0]:
    tab_subtitles = ["Family", "Sample Adult", "Sample Child"]
    subtabs       = st.tabs(tab_subtitles)

    # Common section
    data_ids = lmap(lambda s: s.lower().replace(' ', '_'), tab_subtitles)
    metas, datas = [], []
    for ttt in data_ids:
        meta, data = load_dataset(ttt)
        metas.append(meta)
        datas.append(data)
    
    metas = dict(zip(data_ids, metas))
    datas = dict(zip(data_ids, datas))
    
    with subtabs[0]:
        data_id = data_ids[0]

        show_overview(metas, datas, data_id)
        show_feature_distribution(metas, datas, data_id)

    with subtabs[1]:
        data_id = data_ids[1]

        show_overview(metas, datas, data_id)
        show_feature_distribution(metas, datas, data_id)
        show_target_distribution(metas, datas, data_id, 0)

    with subtabs[2]:
        data_id = data_ids[2]

        show_overview(metas, datas, data_id)
        show_feature_distribution(metas, datas, data_id)
        show_target_distribution(metas, datas, data_id, 1)


with tabs[1]:
    st.header("1. Replace Minor Features with 'Unknown'(-1)")

    tab_subtitles = ["Family", "Sample Adult", "Sample Child"]
    tab_subtitles2 = ["Family-Sample Adult", "Family-Sample Child"]
    subtabs1       = st.tabs(tab_subtitles)

    # Common section
    data_ids = lmap(lambda s: s.lower().replace(' ', '_'), tab_subtitles)
    metas, datas = [], []
    for ttt in data_ids:
        meta, data = load_dataset(ttt)
        metas.append(meta)
        datas.append(data)
    
    metas = dict(zip(data_ids, metas))
    datas = dict(zip(data_ids, datas))

    with subtabs1[0]:
        data_id = data_ids[0]
        show_replace_minor(metas, datas, data_id)
    with subtabs1[1]:
        data_id = data_ids[1]
        show_replace_minor(metas, datas, data_id)
    with subtabs1[2]:
        data_id = data_ids[2]
        show_replace_minor(metas, datas, data_id)

    st.header("2. Impute Nan values to 'Unknown'(-1)")
    subtabs2 = st.tabs(tab_subtitles)
    with subtabs2[0]:
        data_id = data_ids[0]
        show_impute_data(metas, datas, data_id)
    with subtabs2[1]:
        data_id = data_ids[1]
        show_impute_data(metas, datas, data_id)
    with subtabs2[2]:
        data_id = data_ids[2]
        show_impute_data(metas, datas, data_id)

    st.header("3. Set Data types of Features (numerical / categorical)")
    subtabs3 = st.tabs(tab_subtitles)
    with subtabs3[0]:
        data_id = data_ids[0]
        show_set_dtypes(metas, datas, data_id)
    with subtabs3[1]:
        data_id = data_ids[1]
        show_set_dtypes(metas, datas, data_id)
    with subtabs3[2]:
        data_id = data_ids[2]
        show_set_dtypes(metas, datas, data_id)

    st.header("4. Impute Numerical Features (mode)")
    subtabs4 = st.tabs(tab_subtitles)
    with subtabs4[0]:
        data_id = data_ids[0]
        show_impute_numerical_features(metas, datas, data_id)
    with subtabs4[1]:
        data_id = data_ids[1]
        show_impute_numerical_features(metas, datas, data_id)
    with subtabs4[2]:
        data_id = data_ids[2]
        show_impute_numerical_features(metas, datas, data_id)

    st.header("5. Get Feature / Target")
    subtabs5 = st.tabs(tab_subtitles)
    with subtabs5[0]:
        data_id = data_ids[0]
        show_extract_features(metas, datas, data_id)
    with subtabs5[1]:
        data_id = data_ids[1]
        show_extract_features(metas, datas, data_id, target='DIBEV1')
    with subtabs5[2]:
        data_id = data_ids[2]
        show_extract_features(metas, datas, data_id, target='CCONDRR6')

    st.header("6. Drop Columns")
    subtabs6 = st.tabs(tab_subtitles)
    with subtabs6[0]:
        data_id = data_ids[0]
        show_drop_columns(metas, datas, data_id)
    with subtabs6[1]:
        data_id = data_ids[1]
        show_drop_columns(metas, datas, data_id)
    with subtabs6[2]:
        data_id = data_ids[2]
        show_drop_columns(metas, datas, data_id)

    st.header("7. Manual Handling")
    subtabs7 = st.tabs(tab_subtitles)
    with subtabs7[0]:
        data_id = data_ids[0]
        show_manual_handling(metas, datas, data_id)
    with subtabs7[1]:
        data_id = data_ids[1]
        show_manual_handling(metas, datas, data_id)
    with subtabs7[2]:
        data_id = data_ids[2]
        show_manual_handling(metas, datas, data_id)

    st.header("8. Drop Diabetes-relevant Columns")
    subtabs8 = st.tabs(tab_subtitles)
    with subtabs8[0]:
        data_id = data_ids[0]
        show_drop_diabetes_columns(metas, datas, data_id)
    with subtabs8[1]:
        data_id = data_ids[1]
        show_drop_diabetes_columns(metas, datas, data_id)
    with subtabs8[2]:
        data_id = data_ids[2]
        show_drop_diabetes_columns(metas, datas, data_id)
        
    st.header("9. Standard Scaler")
    subtabs9 = st.tabs(tab_subtitles)
    with subtabs9[0]:
        data_id = data_ids[0]
        show_standard_scaler(metas, datas, data_id)
    with subtabs9[1]:
        data_id = data_ids[1]
        show_standard_scaler(metas, datas, data_id)
    with subtabs9[2]:
        data_id = data_ids[2]
        show_standard_scaler(metas, datas, data_id)
        
    with open(PATH.final, "wb") as f:
        pickle.dump(metas, f)

    st.header("10. Correlation Coefficients with Target")
    subtabs10 = st.tabs(tab_subtitles2)
    with subtabs10[0]:
        show_correlations(metas, 0)
    with subtabs10[1]:
        show_correlations(metas, 1)
        
    # st.header("11. Drop Columns based on VIF > 10")
    # subtabs11 = st.tabs(tab_subtitles2)
    # with subtabs11[0]:
    #     show_vif(0)
    # with subtabs11[1]:
    #     show_vif(1)


with tabs[2]:
    tab_subtitles = ["Family-Sample Adult", "Family-Sample Child"]
    subtabs = st.tabs(tab_subtitles)
    
    with subtabs[0]:
        st.header("1. Data Selection")
        st.write("#### 1.1 Merge Family and Sample Adult Dataset")
        data, meta = select_dataset(0)
        st.dataframe(data)

        st.write("#### 1.2 Split Dataset")
        dataset = split_dataset(data, 0)

        st.header("2. Logistic Regression (5 Folds)")
        show_logistic_regression(data, meta, dataset, 0)

        st.header("3. Random Forest Classifier (5 Folds)")
        show_random_forest_classifier(meta, dataset)
        
        st.header("4. LightGBM Classifier (5 Folds)")
        show_lightgbm_classifier(meta, dataset)
        
    with subtabs[1]:
        st.header("1. Data Selection")
        st.write("#### 1.1 Merge Family and Sample Child Dataset")
        data, meta = select_dataset(1)
        st.dataframe(data)

        st.write("#### 1.2 Split Dataset")
        dataset = split_dataset(data, 1)

        st.header("2. Logistic Regression (5 Folds Soft Voting Ensemble)")
        show_logistic_regression(data, meta, dataset, 1)

        st.header("3. Random Forest Classifier (5 Folds Soft Voting Ensemble)")
        show_random_forest_classifier(meta, dataset)
        
        st.header("4. LightGBM Classifier (5 Folds Soft Voting Ensemble)")
        show_lightgbm_classifier(meta, dataset)
