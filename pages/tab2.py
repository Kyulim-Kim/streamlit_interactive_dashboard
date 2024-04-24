from utils.app.io import *
from utils.data.dataloader import *
from utils.data.preprocessing import *
from utils.data.metrics import *
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle


get_score = lambda y, p: dict(f1_score=f1_score(y, p, average='macro'), accuracy=accuracy_score(y, p))
select_num_cols = lambda corr, threshold, last: corr[(corr['corr_abs'].cumsum() / last) <= threshold].index if threshold < 1 else corr.index[:threshold]
select_cat_cols = lambda fis, threshold: fis[fis['feature_importance'].cumsum() <= threshold].dropna().index if threshold < 1 else fis.index[:threshold]
select_cat_cols2 = lambda fis, threshold, last: fis[(fis['feature_importance'].cumsum() / last) <= threshold].dropna().index if threshold < 1 else fis.index[:threshold]

def training(base_estimator, X_tv, y_tv, n_splits=5, verbose=False):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=PARAMS.seed)

    models = []
    scores = []
    for idxs_train, idxs_test in cv.split(X_tv, y_tv):
        clone_model = clone(base_estimator)
        X_train_fold, y_train_fold = X_tv.loc[idxs_train], y_tv.loc[idxs_train]
        X_val_fold, y_val_fold = X_tv.loc[idxs_test], y_tv.loc[idxs_test]

        clone_model.fit(X_train_fold, y_train_fold)
        p_train_fold = clone_model.predict(X_train_fold)
        p_val_fold = clone_model.predict(X_val_fold)

        train_score, val_score = get_score(y_train_fold, p_train_fold), get_score(y_val_fold, p_val_fold)
        score = {'train_acc': train_score['accuracy'], 'train_f1': train_score['f1_score'],
                 'val_acc': val_score['accuracy'], 'val_f1': val_score['f1_score']}
        if verbose:
            print(score)

        models.append(clone_model)
        scores.append(score)

    scores = pd.DataFrame(scores).T
    scores = pd.concat([scores, scores.mean(axis=1)], axis=1)
    scores.columns = [f'fold_{i}' for i in range(5)] + ['mean']
    return models, scores


def test(models, X_test, y_test):
    proba_test = np.array([model.predict_proba(X_test) for model in models]).mean(axis=0)
    p_test = proba_test.argmax(axis=1)
    scores = get_score(y_test, p_test)
    print("Test score(macro):", scores)
    return scores


def select_dataset(d):
    # metas, datas = load_dataset()
    with open(PATH.final, "rb") as f:
        metas = pickle.load(f)
    family       = pd.read_feather(PATH.family_final_data)
    sample_adult = pd.read_feather(PATH.sample_adult_final_data)
    sample_child = pd.read_feather(PATH.sample_child_final_data)
    # smote = SMOTE(random_state=42)

    if d == 0:
        target = PARAMS.target
        data = pd.merge(sample_adult, family, how='left', on='family_id')
        data = data[data[target].isin([1, 2])]
        meta = pd.concat([metas['family'], metas['sample_adult']])
        
    if d == 1:
        target = PARAMS.target2
        data = pd.merge(sample_child, family, how='left', on='family_id')
        data = data[data[target].isin([1, 2])]
        meta = pd.concat([metas['family'], metas['sample_child']])
    
    # X_data = data.drop(columns=target)
    # y_data = data[target]    
    
    # X_data, y_data = smote.fit_resample(X_data, y_data)
    # resampled = pd.DataFrame(X_data, columns=X_data.columns)
    # resampled[target] = y_data
    
    # with st_stdout("code"):
    #     print("Target distribution before SMOTE oversampling:\n")
    #     print(data[target].value_counts())
    #     print("Target distribution after SMOTE oversampling:\n")
    #     print(resampled[target].value_counts())
        
    return data, meta


def split_dataset(data, d):
    if d == 0:
        target = PARAMS.target
    elif d == 1:
        target = PARAMS.target2
    train_val_data, test_data = train_test_split(data, test_size=0.2, stratify=data[target], random_state=PARAMS.seed)
    train_val_data.reset_index(drop=True, inplace=True)

    X_tv = train_val_data.drop(columns=target)
    y_tv = train_val_data[target]

    X_test = test_data.drop(columns=target)
    y_test = test_data[target]

    with st_stdout("code"):
        print(f"Train + validation data: {len(train_val_data)}")
        print(f"Test data: {len(test_data)}")

    return dict(
        X_tv=X_tv, y_tv=y_tv,
        X_test=X_test, y_test=y_test
    )


def show_logistic_regression(data, meta, dataset, d):    
    n_cols  = 30
    corr    = get_corr(meta, data, d)
    last = corr['corr_abs'].cumsum().iloc[-1]
    df_desc = get_description(meta, corr.index[:n_cols])
    fig, ax = plt.subplots(figsize=(30, 3))
    corr['corr_abs'][:n_cols].plot.bar(title=f"abs(Correlation Coefficient) (top {n_cols} features from {len(corr)} features)", ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(df_desc)

    st.write("#### Feature Selection using Cumulative Sum of Correlation Coefficients")
    for threshold in [0.9, 0.6, 0.3, 0.2, 0.1, 0.05]:
        cols = select_num_cols(corr, threshold, last)
        model = LogisticRegression(random_state=PARAMS.seed)
        models, train_val_scores = training(model, dataset['X_tv'][cols], dataset['y_tv'])
        test_scores = test(models, dataset['X_test'][cols], dataset['y_test'])

        with st_stdout("code"):
            print(f"Threshold: {threshold} (# columns: {len(cols)})")
            print(f"Test score(macro): {test_scores}")
        st.dataframe(train_val_scores)
        

def show_random_forest_classifier(meta, dataset):
    model = RandomForestClassifier(random_state=PARAMS.seed, n_jobs=-1)
    models, train_val_scores = training(model, dataset['X_tv'], dataset['y_tv'])
    fis = pd.DataFrame(np.mean([model.feature_importances_ for model in models], axis=0),
                       index=models[0].feature_names_in_, columns=['feature_importance'])
    fis.sort_values('feature_importance', ascending=False, inplace=True)

    n_cols = 30
    df_desc = get_description(meta, fis.index[:n_cols])
    fig, ax = plt.subplots(figsize=(30, 3))
    fis.head(n_cols).plot.bar(legend=False, title=f"Feature importances from RandomForestClassifier (average of 5 models, {n_cols} features from {len(fis)} features)", ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(df_desc)

    st.write("#### Feature Selection using Cumulative Sum of GINI Importance")
    for threshold in [0.9, 0.6, 0.3, 0.2, 0.1, 0.05]:
        cols  = select_cat_cols(fis, threshold)
        model = RandomForestClassifier(random_state=PARAMS.seed, n_jobs=-1)
        models, train_val_scores = training(model, dataset['X_tv'][cols], dataset['y_tv'])
        test_scores = test(models, dataset['X_test'][cols], dataset['y_test'])

        with st_stdout("code"):
            print(f"Threshold: {threshold} (# columns: {len(cols)})")
            print(f"Test score(macro): {test_scores}")
        st.dataframe(train_val_scores)
        

def show_lightgbm_classifier(meta, dataset):
    model = lgb.LGBMClassifier(random_state=PARAMS.seed, n_jobs=-1)
    label_encoder = LabelEncoder()
    categorical1 = [col for col, dtype in dataset['X_tv'].dtypes.items() if dtype == 'object']
    categorical2 = [col for col, dtype in dataset['X_test'].dtypes.items() if dtype == 'object']
    for col in categorical1:
        dataset['X_tv'][col] = label_encoder.fit_transform(dataset['X_tv'][col])
    for col in categorical2:
        dataset['X_test'][col] = label_encoder.fit_transform(dataset['X_test'][col])
    lgb_train = lgb.Dataset(data=dataset['X_tv'], label=dataset['y_tv'])
    lgb_test = lgb.Dataset(data=dataset['X_test'], label=dataset['y_test'])
    models, train_val_scores = training(model, lgb_train.data, lgb_train.label)
    fis = pd.DataFrame(np.mean([model.feature_importances_ for model in models], axis=0),
                       index=models[0].feature_name_, columns=['feature_importance'])
    fis.sort_values('feature_importance', ascending=False, inplace=True)
    last = fis['feature_importance'].cumsum().iloc[-1]

    n_cols = 30
    df_desc = get_description(meta, fis.index[:n_cols])
    fig, ax = plt.subplots(figsize=(30, 3))
    fis.head(n_cols).plot.bar(legend=False, title=f"Feature importances from LGBMClassifier (average of 5 models, {n_cols} features from {len(fis)} features)", ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(df_desc)

    st.write("#### Feature Selection using Cumulative Sum of Split Importance")
    for threshold in [0.9, 0.6, 0.3, 0.2, 0.1, 0.05]:
        cols  = select_cat_cols2(fis, threshold, last)
        model = lgb.LGBMClassifier(random_state=PARAMS.seed, n_jobs=-1)
        models, train_val_scores = training(model, lgb_train.data[cols], lgb_train.label)
        test_scores = test(models, lgb_test.data[cols], lgb_test.label)

        with st_stdout("code"):
            print(f"Threshold: {threshold} (# columns: {len(cols)})")
            print(f"Test score(macro): {test_scores}")
        st.dataframe(train_val_scores)
