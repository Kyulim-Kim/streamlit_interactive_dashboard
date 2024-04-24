from _utils import *
from utils.data.preprocessing import *

from io import StringIO
from threading import current_thread
from contextlib import contextmanager

import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import SCRIPT_RUN_CONTEXT_ATTR_NAME


def run_command(*args, **kwargs):
    result = subprocess.Popen(stdout=subprocess.PIPE, *args, **kwargs)
    for line in result.stdout:
        print(line.decode('utf-8'))


# Reference: https://discuss.streamlit.io/t/cannot-print-the-terminal-output-in-streamlit/6602/9
@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield

# @contextmanager
# def st_stderr(dst):
#     with st_redirect(sys.stderr, dst):
#         yield


def show_numerical_features(metas, datas, data_id):
    pp = Preprocessor(metas)
    pp.fit = True
    
    meta, data = metas[data_id], datas[data_id]
    data = pp.set_dtypes(data, meta)

    data_num = data.select_dtypes(exclude = 'object')
    n_features, n_features_sample = len(data_num.columns), 5
    n_rows, n_samples = len(data_num), 5000

    cols    = np.random.choice(data_num.columns, n_features_sample, replace=False)
    sample  = data_num.sample(n_samples)[cols]
    df_desc = meta[meta['final_id'].isin(cols)].drop(columns='question_id')

    title = f"Numerical Features (sampling: {n_features_sample} features from {n_features} features, {n_samples} rows from {n_rows} rows)"
    fig, axes = plt.subplots(1, n_features_sample, figsize=PARAMS.figsize)
    for idx, ax in enumerate(axes.flat):
        col = df_desc['final_id'].values[idx]
        sns.histplot(sample[col], kde=True, bins=100, ax=ax)
        if ax != axes.flat[0]:
            ax.set_ylabel(None)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(df_desc)

def show_categorical_features(metas, datas, data_id):
    pp = Preprocessor(metas)
    pp.fit = True
    
    meta, data = metas[data_id], datas[data_id]
    data = pp.set_dtypes(data, meta)

    data_cat = data.select_dtypes(include = 'object')
    n_features, n_features_sample = len(data_cat.columns), 5
    n_rows, n_samples = len(data_cat), 5000

    cols    = np.random.choice(data_cat.columns, n_features_sample, replace=False)
    sample  = data_cat.sample(n_samples)[cols]
    df_desc = meta[meta['final_id'].isin(cols)].drop(columns='question_id')

    title = f"Categorical Features (sampling: {n_features_sample} features from {n_features} features, {n_samples} rows from {n_rows} rows)"
    fig, axes = plt.subplots(1, n_features_sample, figsize=PARAMS.figsize)
    for idx, ax in enumerate(axes.flat):
        col = df_desc['final_id'].values[idx]
        sns.countplot(data=sample, x=col, ax=ax)
        if ax != axes.flat[0]:
            ax.set_ylabel(None)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(df_desc)

def get_description(meta, final_ids):
    final_ids = [id for id in final_ids if id in meta['final_id'].values]
    return meta.set_index('final_id').loc[final_ids][['description', 'options', 'keywords']]

def show_target_eda(metas, datas, data_id, d):
    pp = Preprocessor(metas)
    pp.fit = True
    
    meta, data = metas[data_id], datas[data_id]
    data = pp.set_dtypes(data, meta)

    if d == 0:
        target = PARAMS.target
        data['BMI'] = data['BMI'].apply(lambda x: 2000 if x < 2000 else (4500 if x > 4500 else ((x // 100) * 100)))
        data['AGE_P'] = data['AGE_P'].apply(lambda x: 1 if x < 10 else (85 if x > 84 else ((x // 10) * 10)))
        cols = [target, 'SEX', 'BMI', 'AGE_P', 'REGION']
    elif d == 1:
        target = PARAMS.target2
        data['BMI_SC'] = data['BMI_SC'].apply(lambda x: 1500 if x < 1500 else (4500 if x > 4500 else ((x // 100) * 100)))
        data['AGE_P'] = pd.cut(data['AGE_P'], bins=range(0, 18, 1)).apply(lambda x: x.left + 1)
        cols = [target, 'SEX', 'BMI_SC', 'AGE_P', 'REGION']
        
    data_eda = data[cols]
    n_features_sample = len(cols)
    df_desc = meta[meta['final_id'].isin(cols)].drop(columns='question_id')
    df_desc = df_desc.set_index('final_id').reindex(cols)
    
    subset_yes = data_eda[data_eda[target] == 1]
    subset_no = data_eda[data_eda[target] == 2]
    
    fig, axes = plt.subplots(2, n_features_sample, figsize=(28, 10))
    title = f"With(top) / Without(bottom) Diabetes (sampling: {n_features_sample} features)"
    fig.suptitle(title, fontsize=15)
    for axes_row, df in zip(axes, [subset_yes, subset_no]):
        for ax, col in zip(axes_row.flat, cols):
            count = df[col].value_counts()
            ax.pie(count, labels=count.index, autopct='%1.1f%%', startangle=140)
            if ax != axes_row[0]:
                ax.set_ylabel(None)
    for ax in axes[0]:
        ax.set_xlabel(None)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(df_desc)
