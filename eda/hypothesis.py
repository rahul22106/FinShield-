import pandas as pd
from scipy import stats
from utils.logger import get_logger

logger = get_logger(__name__)


def test_amount_vs_fraud(df: pd.DataFrame) -> dict:
    """
    Test 1 — Mann-Whitney U Test
    H0: No difference in amount between fraud and legit
    """
    legit = df[df['Class'] == 0]['amount']
    fraud = df[df['Class'] == 1]['amount']

    stat, p_value = stats.mannwhitneyu(
        fraud, legit, alternative='two-sided'
    )

    result = {
        'test':      'Mann-Whitney U',
        'variable':  'Transaction Amount',
        'statistic': round(stat, 4),
        'p_value':   round(p_value, 6),
        'passed':    p_value < 0.05,
        'finding':   'Fraud amounts differ significantly from legit'
                     if p_value < 0.05 else
                     'No significant difference in amounts'
    }

    status = "SIGNIFICANT" if result['passed'] else "NOT SIGNIFICANT"
    logger.info(f"Test 1 - Amount vs Fraud: {status} (p={p_value:.6f})")
    return result


def test_time_vs_fraud(df: pd.DataFrame) -> dict:
    """
    Test 2 — Chi-Square Test
    H0: Fraud is independent of day of week
    """
    contingency = pd.crosstab(df['day_of_week'], df['Class'])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    result = {
        'test':      'Chi-Square',
        'variable':  'Day of Week',
        'statistic': round(chi2, 4),
        'p_value':   round(p_value, 6),
        'passed':    p_value < 0.05,
        'finding':   'Fraud significantly depends on day of week'
                     if p_value < 0.05 else
                     'Fraud is independent of day of week'
    }

    status = "SIGNIFICANT" if result['passed'] else "NOT SIGNIFICANT"
    logger.info(f"Test 2 - Time vs Fraud: {status} (p={p_value:.6f})")
    return result


def test_amount_range_vs_fraud(df: pd.DataFrame) -> dict:
    """
    Test 3 — Chi-Square Test
    H0: Fraud rate is same across amount ranges
    """
    contingency = pd.crosstab(df['amount_range'], df['Class'])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    result = {
        'test':      'Chi-Square',
        'variable':  'Amount Range',
        'statistic': round(chi2, 4),
        'p_value':   round(p_value, 6),
        'passed':    p_value < 0.05,
        'finding':   'Fraud rate varies significantly by amount range'
                     if p_value < 0.05 else
                     'No significant variation across amount ranges'
    }

    status = "SIGNIFICANT" if result['passed'] else "NOT SIGNIFICANT"
    logger.info(f"Test 3 - Amount Range vs Fraud: {status} (p={p_value:.6f})")
    return result


def test_weekend_vs_fraud(df: pd.DataFrame) -> dict:
    """
    Test 4 — Chi-Square Test
    H0: Fraud rate is same on weekdays vs weekends
    """
    contingency = pd.crosstab(df['is_weekend'], df['Class'])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    result = {
        'test':      'Chi-Square',
        'variable':  'Weekend vs Weekday',
        'statistic': round(chi2, 4),
        'p_value':   round(p_value, 6),
        'passed':    p_value < 0.05,
        'finding':   'Fraud rate differs on weekends vs weekdays'
                     if p_value < 0.05 else
                     'No significant difference on weekends vs weekdays'
    }

    status = "SIGNIFICANT" if result['passed'] else "NOT SIGNIFICANT"
    logger.info(f"Test 4 - Weekend vs Fraud: {status} (p={p_value:.6f})")
    return result


def run_hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run all hypothesis tests and return summary."""
    logger.info("Starting hypothesis tests...")
    logger.info("-" * 50)

    results = [
        test_amount_vs_fraud(df),
        test_time_vs_fraud(df),
        test_amount_range_vs_fraud(df),
        test_weekend_vs_fraud(df),
    ]

    summary = pd.DataFrame(results)

    logger.info("-" * 50)
    logger.info("Hypothesis Testing Summary:")
    for _, row in summary.iterrows():
        status = "REJECT H0" if row['passed'] else "FAIL TO REJECT H0"
        logger.info(f"  {row['test']} | {row['variable']} | {status}")
        logger.info(f"  Finding: {row['finding']}")
        logger.info("")

    return summary