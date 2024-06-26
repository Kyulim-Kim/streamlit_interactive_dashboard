import pandas as pd
from _utils import *
from statsmodels.stats.outliers_influence import variance_inflation_factor
from dask import delayed, compute
from dask.diagnostics import ProgressBar


@T
def get_corr(metadata: pd.DataFrame, data: pd.DataFrame, d) -> pd.DataFrame:
    """Get correlation of `data` with target column.

    Args:
        metadata: Metadata dataframe.
        data: Dataframe.

    Returns:
        Correlation dataframe.
    """
    if d == 0:
        target = PARAMS.target
    elif d == 1:
        target = PARAMS.target2
    assert target in data, f"Data should have {target} column"

    # Get correlation
    C = data.select_dtypes(exclude='object').corr()
    corr = pd.DataFrame([C[target], C[target].abs()], index=['corr', 'corr_abs']).T
    corr.index.name = 'final_id'
    corr.sort_values('corr_abs', ascending=False, inplace=True)
    corr = corr.iloc[1:]

    # Merge with metadata
    corr = pd.merge(corr, metadata.set_index('final_id'), how='left', on='final_id')
    return corr[['corr_abs', 'corr', 'description', 'options']]


@T
def get_VIF(data: pd.DataFrame, plot: bool = False, d=None) -> pd.DataFrame:
    """Get variance inflation factor.

    Args:
        data: Dataframe.
        plot: Whether to plot or not.

    Returns:
        Variance inflation factor.
    """
    
    if d == 0:
        target = PARAMS.target
    elif d == 1:
        target = PARAMS.target2

    num_data = data.select_dtypes(exclude='object')
    if target in num_data:
        num_data.drop(columns=target, inplace=True)

    rst = pd.DataFrame(index=num_data.columns)
    with ProgressBar():
        tasks = [delayed(variance_inflation_factor)(num_data.values, i) for i in range(len(num_data.columns))]
        rst['VIF'] = compute(*tasks, scheduler='processes')

    if plot:
        plot_VIF(rst)
        ax = rst.plot.bar(figsize=PARAMS.figsize)
        ax.axhline(5, color='k')  # VIF base: 5 or 10
    return rst

def plot_VIF(data=None):
    ax = data.plot.bar(legend=None, figsize=PARAMS.figsize)
    ax.axhline(5, color='k')
    ax.set_title("Variance Inflation Factor")


# def plot_corr(data: pd.DataFrame, cols: list = None):
#     if cols is not None:
#         data = data[cols]
#     corr = data.select_dtypes('number').corr()
#
#     correlation matrix
#     fig, ax = plt.subplots(figsize=PARAMS.figsize)
#     ax.set_title(f"Correlation Matrix (C ≥ 0.4, max: {np.max(corr[corr != 1]):.2f})")
#     mask = np.zeros_like(corr)
#     mask[(corr < 0.4) | (corr == 1)] = True
#     sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.1f', center=0, ax=ax, cbar=False)
#
#     # mask = np.zeros_like(corr)
#     # mask[np.triu_indices_from(mask)] = True
#     # sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.1f', center=0, ax=ax, cbar=False)
