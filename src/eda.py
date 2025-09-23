"""
Reusable uncions for Exploratory Data Analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, shapiro, kstest, normaltest, probplot
from fitter import Fitter
from IPython.display import display

#=============================================================================#

def distribution_analysis(pandas_series: pd.Series, name: str):
    """
    Performs a comprehensive analysis of the distribution of a 
    given variable in pandas.Series format.

    Parameters:
        pandas_series (pd.Series): The data series to analyze.
        name (str): The name of the variable (for labeling).

    Returns:
        pd.DataFrame: Summary of key results (analysis, result, interpretation).

    Analysis: 
        1. Visualization: Histogram + KDE, Boxplot, QQ-Plot
        2. Skewness and Kurtosis
        3. Normality Tests
        4. Fit to common distributions
        5. Fit to all Fitter distributions
    """
    
    print(f"\n{'='*20} Distribution Analysis for: {name} {'='*20}\n")

     # Drop missing values
    data = pandas_series.dropna()

    # 1. Visualization: Histogram + KDE, Boxplot, QQ-Plot =====================
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram ---------------------------------
    axs[0].hist(data, bins=50, density=False, alpha=0.7, 
                color='skyblue', edgecolor='black')
    axs[0].set_title(f'{name} - Histogram')
    axs[0].set_xlabel(name)
    axs[0].set_ylabel('Frequency')
    
    # Box-Plot ----------------------------------
    axs[1].boxplot(data, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='lightgreen', color='black'),
                   medianprops=dict(color='red'))
    axs[1].set_title(f'{name} - Boxplot')
    axs[1].set_xlabel(name)

    # Q-Q plot ----------------------------------
    probplot(data, dist="norm", plot=axs[2])
    axs[2].set_title(f'{name} - QQ Plot')

    lines = axs[2].get_lines()
    if len(lines) >= 2:
        lines[0].set_markerfacecolor('skyblue')
        lines[0].set_markeredgecolor('skyblue')
        lines[1].set_color('red')

    plt.tight_layout()
    plt.show()

    # 2. Skewness and Kurtosis ================================================
    skew_val = skew(data)
    kurt_val = kurtosis(data)

    print('='*80)
    print("Skewness and Kurtosis:")
    
    interpretation_skew = "-> Positive skew (right-tailed)" if skew_val > 0 else (
        "-> Negative skew (left-tailed)" if skew_val < 0 else "-> Symmetric distribution")
    
    interpretation_kurt = "-> Leptokurtic (heavy tails)" if kurt_val > 3 else (
        "-> Platykurtic (light tails)" if kurt_val < 3 else "-> Mesokurtic (normal-like)")
    
    print(f"-Skewness  : {skew_val:.8f} {interpretation_skew}")
    print(f"-Kurtosis  : {kurt_val:.8f} {interpretation_kurt}")

    # 3. Normality Tests ======================================================
    print()
    print('='*80)
    print("Normality Tests:")
    
    # Shapiro-Wilk (limit 5000) -----------------
    if len(data) > 5000:
        print(f"Shapiro-Wilk: sample size is {len(data)} > 5000, using random sample of 5000.")
        sample = data.sample(5000, random_state=1)
    else:
        sample = data

    shapiro_stat, shapiro_p = shapiro(sample)

    # Kolmogorov-Smirnov ------------------------
    ks_stat, ks_p = kstest(data, 'norm', args=(np.mean(data), np.std(data)))

    # D'Agostino --------------------------------
    dagostino_stat, dagostino_p = normaltest(data)

    def interpret_p(p):
        return "p < 0.05: reject H₀ (not normal)" if p < 0.05 else "p ≥ 0.05: fail to reject H₀ (possibly normal)"

    interp_shapiro = interpret_p(shapiro_p)
    interp_ks = interpret_p(ks_p)
    interp_dagostino = interpret_p(dagostino_p)

    print()
    print(f"- Shapiro-Wilk p-value      : {shapiro_p:.8f} -> {interp_shapiro}")
    print(f"- Kolmogorov-Smirnov p-value: {ks_p:.8f} -> {interp_ks}")
    print(f"- D’Agostino-Pearson p-value: {dagostino_p:.8f} -> {interp_dagostino}")

    # 4. Fit to common distributions ==========================================
    print('='*80)
    print("Fitting Data to Distributions:")
    print("1. Fitting to common distributions:")
    f_common = Fitter(data,
                      distributions=['norm', 'lognorm', 'gamma',
                                     'expon', 'beta', 'chi', 'chi2', 'logistic'],
                      timeout=10)

    f_common.fit()
    f_common_summary = f_common.summary()
    plt.show()
    display(f_common_summary)

    # Best
    common_best = f_common.get_best(method='sumsquare_error')
    common_best_name = list(common_best.keys())[0]

    # 5. Fit to all distributions =============================================
    print("\n2. Fitting to all distributions (this may take some time)...")
    f_all = Fitter(data, timeout=20)
    f_all.fit()
    f_all_summary = f_all.summary()
    plt.show()
    display(f_all_summary)

    # Best
    all_best = f_all.get_best(method='sumsquare_error')
    all_best_name = list(all_best.keys())[0]


    # 6. Info Explanation =====================================================
    info_text = """
    Information:
    - sumsquare_error: Lower means better fit to data.
    - aic (Akaike Information Criterion): Lower is better, penalizes overfitting.
    - bic (Bayesian Information Criterion): Same as AIC but stronger penalty on complexity.
    - kl_div (Kullback-Leibler divergence): Distance between true and fitted distributions, closer to 0 is better.
    - ks_statistic / ks_pvalue: Kolmogorov-Smirnov test statistic and p-value.
        """
    print(info_text)

    # 7. Summary Output Table =================================================
    summary_data = [
        ["Skewness", round(skew_val, 6), interpretation_skew.replace("-> ", "")],
        ["Kurtosis", round(kurt_val, 6), interpretation_kurt.replace("-> ", "")],
        ["Normality Test - Shapiro Wilk", f"{shapiro_p:.6f}", interp_shapiro],
        ["Normality Test - Kolmogorov-Smirnov", f"{ks_p:.6f}", interp_ks],
        ["Normality Test - D’Agostino", f"{dagostino_p:.6f}", interp_dagostino],
        ["Best Fit (common)", common_best_name, '-'],
        ["Best Fit (all)", all_best_name, '-']
    ]
    summary_df = pd.DataFrame(summary_data, columns=["Analysis", "Result", "Interpretation"])

    return summary_df


#=============================================================================#
def distribution_analysis_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform distribution analysis on all numeric columns in the DataFrame.
    
    Returns a DataFrame with skewness, kurtosis, normality test p-values,
    and best-fitting distributions.
    """

    def interpret_skew(val):
        if val > 0:
            return "Positive skew (right-tailed)"
        elif val < 0:
            return "Negative skew (left-tailed)"
        else:
            return "Symmetric distribution"

    def interpret_kurt(val):
        if val > 3:
            return "Leptokurtic (heavy tails)"
        elif val < 3:
            return "Platykurtic (light tails)"
        else:
            return "Mesokurtic (normal-like)"

    def interpret_p(p):
        return "p < 0.05: reject H₀ (not normal)" if p < 0.05 else "p ≥ 0.05: fail to reject H₀ (possibly normal)"

    results = []

    numeric_df = df.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        data = numeric_df[col].dropna()

        if len(data) < 8:
            continue  # skip very small samples

        # Skewness and Kurtosis
        skew_val = skew(data)
        kurt_val = kurtosis(data)

        # Normality tests
        sample = data.sample(5000, random_state=1) if len(data) > 5000 else data

        shapiro_p = shapiro(sample)[1]
        ks_p = kstest(data, 'norm', args=(np.mean(data), np.std(data)))[1]
        dagostino_p = normaltest(data)[1]

        # Fit to common distributions
        f_common = Fitter(
            data,
            distributions=['norm', 'lognorm', 'gamma', 'expon', 'beta', 'chi', 'chi2', 'logistic'],
            timeout=10
        )
        f_common.fit()
        best_common = list(f_common.get_best().keys())[0]

        results.append({
            "variable": col,
            "skewness": round(skew_val, 6),
            "skewness_interpretation": interpret_skew(skew_val),
            "kurtosis": round(kurt_val, 6),
            "kurtosis_interpretation": interpret_kurt(kurt_val),
            "shapiro_wilk": round(shapiro_p, 6),
            "shapiro_wilk_interpret": interpret_p(shapiro_p),
            "kolmogorov-smirnov": round(ks_p, 6),
            "kolmogorov-smirnov_interpret": interpret_p(ks_p),
            "d_agostino": round(dagostino_p, 6),
            "d_agostino_interpret": interpret_p(dagostino_p),
            "best_fit_common": best_common
        })

    return pd.DataFrame(results)

#=============================================================================#