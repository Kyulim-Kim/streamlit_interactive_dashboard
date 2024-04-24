from utils.app.io import *
from utils.data.dataloader import *
from utils.data.preprocessing import *
from utils.data.metrics import *
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _compare_datas(before, after, meta, cols, title):
    df_desc = meta.drop(columns=['question_id', 'keywords'])

    fig, axes = plt.subplots(2, 5, figsize=PARAMS.figsize)
    fig.suptitle(title, fontsize=15)
    for axes_row, df in zip(axes, [before, after]):
        for ax, col in zip(axes_row.flat, cols):
            if col in df.select_dtypes(exclude = 'object'):
                sns.histplot(df[col], kde=True, bins=100, ax=ax)
            else:
                sns.countplot(df, x=col, ax=ax)
            if ax != axes_row[0]:
                ax.set_ylabel(None)
    for ax in axes[0]:
        ax.set_xlabel(None)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    # st.dataframe(df_desc['final_id'])


def show_replace_minor(metas, datas, data_id):
    pp = Preprocessor(metas)
    pp.fit = True
    meta, data = metas[data_id], datas[data_id]

    before = data.copy()
    data, meta = pp.replace_ambiguous_options(data, meta)
    after = data.copy()

    diff = (before.nunique() - after.nunique()) > 0
    cols = diff[diff].index[:5]

    st.write(f"- Replace ambiguous options (""don't know / not ascertained / refused / not available / time period format / undefined / undefinable"") with 'unknown'")
    _compare_datas(before, after, meta, cols, "Replace data (top: before / bottom: after)")
    
    metas[data_id], datas[data_id] = meta, data


def show_impute_data(metas, datas, data_id):
    pp = Preprocessor(metas)
    meta, data = metas[data_id], datas[data_id]

    before = data.copy()
    data = pp.impute_data(data, meta)
    after = data.copy()

    idxs = before.isna().any()
    cols = np.random.choice(idxs[idxs].index, 5, replace=False)

    st.write(f"- Impute nan values with 'unknown' value")
    _compare_datas(before, after, meta, cols, "Impute data (top: before / bottom: after)")
    
    metas[data_id], datas[data_id] = meta, data


def show_set_dtypes(metas, datas, data_id):
    pp = Preprocessor(metas)
    meta, data = metas[data_id], datas[data_id]
    pp.fit = True
    data = pp.set_dtypes(data, meta)

    st.write(f"- Numerical features: only have '-1'('unknown') or numbers in option")
    show_numerical_features(metas, datas, data_id)
    st.write(f"- Categorical features: only have '-1'('unknown') or encoded categories in option")
    show_categorical_features(metas, datas, data_id)
    
    metas[data_id], datas[data_id] = meta, data


def show_impute_numerical_features(metas, datas, data_id):
    pp = Preprocessor(metas)
    pp.fit = True
    meta, data = metas[data_id], datas[data_id]

    before = data.copy()
    data = pp.impute_numerical_features(data, meta)
    after = data.copy()

    num_cols = data.select_dtypes(exclude='object').columns
    cols = (before[num_cols] == -1).sum().sort_values(ascending=False).index[:5]  # most effective columns
    st.write(f"- Fill numeric unknown values with mode (zero-centered or long tailed distribution)")
    _compare_datas(before, after, meta, cols, "Impute numerical data (top: before / bottom: after)")
    
    metas[data_id], datas[data_id] = meta, data


def show_extract_features(metas, datas, data_id, target=None):
    meta, data = metas[data_id], datas[data_id]
    data['family_id'] = data['HHX'] + data['FMX'] + data['SRVY_YR']

    with st_stdout("code"):
        print("FPX: HHX + FMX + SRVY_YR")
        if target == PARAMS.target:
            print(f"target: {target} - Ever been told that you(adult) have diabetes")
            print(f"\t1: {target} = 'yes'")
            print(f"\t2: {target} = 'no'")
            print(f"\t3: {target} = 'borderline or prediabetes'")
            print(f"\t7: {target} = 'refused'")
            print(f"\t8: {target} = 'not ascertained'")
            print(f"\t9: {target} = 'do not know'")
            
        if target == PARAMS.target2:
            print(f"target: {target} - Ever told sample child had diabetes")
            print(f"\t1: {target} = 'mentioned'")
            print(f"\t2: {target} = 'not mentioned'")
            print(f"\t7: {target} = 'refused'")
            print(f"\t8: {target} = 'not ascertained'")
            print(f"\t9: {target} = 'do not know'")

    metas[data_id], datas[data_id] = meta, data


def show_drop_columns(metas, datas, data_id):
    pp = Preprocessor(metas)
    meta, data = metas[data_id], datas[data_id]
    pp.fit = True

    before = data.copy()
    data = pp.drop_columns(data)

    st.write("- Drop constant columns (contains one value)")
    cnts = before.nunique().sort_values()
    constant_cols = cnts[cnts == 1].index
    st.dataframe(before[constant_cols].nunique())

    st.write("- Drop redundant columns ('HHX', 'FMX', 'INTV_QRT', 'FPX', 'SRVY_YR')")
    redundant_cols = ['HHX', 'FMX', 'INTV_QRT', 'FPX', 'SRVY_YR']
    cols = [col for col in redundant_cols if col in before]
    st.dataframe(before[cols].nunique())
    
    metas[data_id], datas[data_id] = meta, data


def show_manual_handling(metas, datas, data_id):
    pp = Preprocessor(metas)
    meta, data = metas[data_id], datas[data_id]

    before = data.copy()
    data = pp.manual_handling(data, meta)

    st.write("- Handle **'do nothing'** options('~unable to do', 'never') to 0")
    cols = []
    for col in before:
        if col in (PARAMS.target, PARAMS.target2, 'family_id'):
            continue
        options = get_meta_values(meta, col)['options']
        options_inversed = inverse_dict(options)
        for option in options_inversed:
            if re.search("^unable to do ", option) or (option == 'never'):
                cols.append(col)

    _compare_datas(before, data, meta, cols, "Manual handling")
    
    metas[data_id], datas[data_id] = meta, data


def show_drop_diabetes_columns(metas, datas, data_id):
    pp = Preprocessor(metas)
    meta, data = metas[data_id], datas[data_id]

    idxs = meta['keywords'].astype(str).str.contains('diabetes')
    diabetes_cols = meta.loc[idxs[idxs].index, 'final_id']
    diabetes_cols = [col for col in diabetes_cols if col in data]
    data = pp.drop_diabetes_columns(data, meta)

    st.write(f"- Drop columns (contain 'diabetes' related keywords)")
    st.dataframe(meta[meta['final_id'].isin(diabetes_cols)][['final_id', 'description', 'options', 'keywords']])
    
    metas[data_id], datas[data_id] = meta, data
        

def show_standard_scaler(metas, datas, data_id):
    pp = Preprocessor(metas)
    meta, data = metas[data_id], datas[data_id]
    pp.fit = True

    before = data.copy()
    
    if data_id == 'family':
        data = pp.standardize(data)
    elif data_id == 'sample_adult':
        data = pp.standardize(data, target=PARAMS.target)
    elif data_id == 'sample_child':
        data = pp.standardize(data, target=PARAMS.target2)
        
    after = data.copy()
    
    num_cols = before.select_dtypes(exclude='object')
    cols = num_cols.apply(lambda x: skew(x)).sort_values(ascending=False).index[-5:]  # most effective columns
    st.write(f"- Standard Scaling")
    _compare_datas(before, after, meta, cols, "standardize numerical data (top: before / bottom: after)")
        
    metas[data_id], datas[data_id] = meta, data
    
    if data_id == 'family':
        datas[data_id].to_feather(PATH.family_final_data)
    elif data_id == 'sample_adult':
        datas[data_id].to_feather(PATH.sample_adult_final_data)
    elif data_id == 'sample_child':
        datas[data_id].to_feather(PATH.sample_child_final_data)
    

def show_correlations(metas, d):
    # metas, datas = load_dataset()
    family       = pd.read_feather(PATH.family_final_data)
    sample_adult = pd.read_feather(PATH.sample_adult_final_data)
    sample_child = pd.read_feather(PATH.sample_child_final_data)
    
    sample_adult_family = pd.merge(sample_adult, family, how='left', on='family_id')
    sample_child_family = pd.merge(sample_child, family, how='left', on='family_id')

    if d == 0:
        data = sample_adult_family
        meta = pd.concat([metas['family'], metas['sample_adult']])
        corr = get_corr(meta, data, d)
        
    if d == 1:
        data = sample_child_family
        meta = pd.concat([metas['family'], metas['sample_child']])
        corr = get_corr(meta, data, d)

    n_cols = 30
    df_desc = get_description(meta, corr.index[:n_cols])
    fig, ax = plt.subplots(figsize=(30, 3))
    corr['corr_abs'][:n_cols].plot.bar(title=f"abs(Correlation Coefficient) (top {n_cols} features from {len(corr)} features)", ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    st.write("- Top 30 abs(Correlation Coefficient)")
    st.dataframe(df_desc)
    
    
# def show_vif(d):
#     family       = pd.read_feather(PATH.family_final_data)
#     sample_adult = pd.read_feather(PATH.sample_adult_final_data)
#     sample_child = pd.read_feather(PATH.sample_child_final_data)

#     if d == 0:
#         data = pd.merge(sample_adult, family, how='left', on='family_id')
        
#     if d == 1:
#         data = pd.merge(sample_child, family, how='left', on='family_id')
    
#     n_data = data.select_dtypes(exclude='object')    
#     vif = pd.DataFrame()
#     vif['VIF_Factor'] = [variance_inflation_factor(n_data.values, i) for i in range(n_data.shape[1])]
#     vif['Feature'] = n_data.columns
    
#     n_cols = 30
#     vif = vif.sort_values(by='VIF_Factor', ascending=False)
#     vif_cols = vif.iloc[:n_cols]
#     fig, ax = plt.subplots(figsize=(30, 3))
#     ax.bar(vif_cols['Feature'], vif_cols['VIF_Factor'], title=f"VIF (top {n_cols} features from {len(vif)} features)", ax=ax)
#     st.pyplot(fig)
#     plt.close(fig)
    
#     st.write("- Columns to drop where VIF > 10")
#     X_to_drop = vif[vif['VIF_Factor'] >= 10]['Feature']
#     st.dataframe(X_to_drop)
