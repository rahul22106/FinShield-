import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.logger import get_logger

logger = get_logger(__name__)

PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# Style
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Plot fraud vs legit distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count plot
    counts = df['Class'].value_counts()
    axes[0].bar(
        ['Legit', 'Fraud'],
        counts.values,
        color=['#2196F3', '#F44336']
    )
    axes[0].set_title('Transaction Class Distribution')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=['Legit', 'Fraud'],
        colors=['#2196F3', '#F44336'],
        autopct='%1.1f%%',
        startangle=90
    )
    axes[1].set_title('Fraud Percentage')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/01_class_distribution.png', dpi=150)
    plt.close()
    logger.info("Plot saved: 01_class_distribution.png")


def plot_amount_distribution(df: pd.DataFrame) -> None:
    """Plot amount distribution for fraud vs legit."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    legit = df[df['Class'] == 0]['amount']
    fraud = df[df['Class'] == 1]['amount']

    # Histogram
    axes[0].hist(legit, bins=50, alpha=0.7,
                 color='#2196F3', label='Legit')
    axes[0].hist(fraud, bins=50, alpha=0.7,
                 color='#F44336', label='Fraud')
    axes[0].set_title('Amount Distribution')
    axes[0].set_xlabel('Amount ($)')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    # Boxplot
    df_plot = df[['amount', 'Class']].copy()
    df_plot['Class'] = df_plot['Class'].map({0: 'Legit', 1: 'Fraud'})
    sns.boxplot(
        data=df_plot,
        x='Class', y='amount',
        palette={'Legit': '#2196F3', 'Fraud': '#F44336'},
        ax=axes[1]
    )
    axes[1].set_title('Amount Boxplot by Class')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/02_amount_distribution.png', dpi=150)
    plt.close()
    logger.info("Plot saved: 02_amount_distribution.png")


def plot_time_analysis(df: pd.DataFrame) -> None:
    """Plot fraud patterns by day and month."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By day of week
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fraud_by_day = df[df['Class'] == 1].groupby('day_of_week').size()
    legit_by_day = df[df['Class'] == 0].groupby('day_of_week').size()

    x = range(7)
    axes[0].bar(x, legit_by_day.reindex(range(7), fill_value=0),
                label='Legit', color='#2196F3', alpha=0.7)
    axes[0].bar(x, fraud_by_day.reindex(range(7), fill_value=0),
                label='Fraud', color='#F44336', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(days)
    axes[0].set_title('Transactions by Day of Week')
    axes[0].legend()

    # By month
    fraud_by_month = df[df['Class'] == 1].groupby('month').size()
    legit_by_month = df[df['Class'] == 0].groupby('month').size()

    months = range(1, 13)
    axes[1].plot(months,
                 legit_by_month.reindex(months, fill_value=0),
                 marker='o', label='Legit', color='#2196F3')
    axes[1].plot(months,
                 fraud_by_month.reindex(months, fill_value=0),
                 marker='o', label='Fraud', color='#F44336')
    axes[1].set_title('Transactions by Month')
    axes[1].set_xlabel('Month')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/03_time_analysis.png', dpi=150)
    plt.close()
    logger.info("Plot saved: 03_time_analysis.png")


def plot_category_analysis(df: pd.DataFrame) -> None:
    """Plot fraud rate by transaction category."""
    category_fraud = df.groupby('category').agg(
        total=('Class', 'count'),
        fraud=('Class', 'sum')
    ).reset_index()
    category_fraud['fraud_rate'] = (
        category_fraud['fraud'] / category_fraud['total'] * 100
    )
    category_fraud = category_fraud.sort_values(
        'fraud_rate', ascending=False
    ).head(10)

    plt.figure(figsize=(12, 6))
    bars = plt.barh(
        category_fraud['category'],
        category_fraud['fraud_rate'],
        color='#F44336',
        alpha=0.8
    )
    plt.title('Top 10 Categories by Fraud Rate (%)')
    plt.xlabel('Fraud Rate (%)')
    for bar, val in zip(bars, category_fraud['fraud_rate']):
        plt.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/04_category_analysis.png', dpi=150)
    plt.close()
    logger.info("Plot saved: 04_category_analysis.png")


def plot_payment_channel(df: pd.DataFrame) -> None:
    """Plot fraud by payment channel."""
    channel_data = df.groupby(
        ['payment_channel', 'Class']
    ).size().unstack(fill_value=0)

    channel_data.plot(
        kind='bar',
        color=['#2196F3', '#F44336'],
        figsize=(10, 5),
        alpha=0.8
    )
    plt.title('Transactions by Payment Channel')
    plt.xlabel('Payment Channel')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(['Legit', 'Fraud'])
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/05_payment_channel.png', dpi=150)
    plt.close()
    logger.info("Plot saved: 05_payment_channel.png")


def run_eda(df: pd.DataFrame) -> None:
    """Run all EDA plots."""
    logger.info("Starting EDA...")
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_time_analysis(df)
    plot_category_analysis(df)
    plot_payment_channel(df)
    logger.info(f"All plots saved to {PLOT_DIR}/")