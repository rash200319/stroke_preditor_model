import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set(style="whitegrid")

def generate_report_assets(file_path):
    df = pd.read_csv(file_path, index_col=0)
    
    # Create a directory for artifacts if it doesn't exist
    # For this environment, we should probably save them in the current directory first
    img_dir = 'analysis_assets'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # 1. Distribution of Numerical Features
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['age'], bins=30, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Age Distribution')
    
    sns.histplot(df['glucose_level'], bins=30, kde=True, ax=axes[1], color='salmon')
    axes[1].set_title('Glucose Level Distribution')
    
    sns.histplot(df['bmi_value'], bins=30, kde=True, ax=axes[2], color='lightgreen')
    axes[2].set_title('BMI Value Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'numerical_distributions.png'))
    plt.close()

    # 2. Stroke Incidence by Category
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.countplot(x='gender', hue='stroke_event', data=df, ax=axes[0,0], palette='viridis')
    axes[0,0].set_title('Stroke by Gender')
    
    sns.countplot(x='age_group', hue='stroke_event', data=df, ax=axes[0,1], palette='viridis')
    axes[0,1].set_title('Stroke by Age Group')
    
    sns.countplot(x='smoking_habit', hue='stroke_event', data=df, ax=axes[1,0], palette='viridis')
    axes[1,0].set_title('Stroke by Smoking Habit')
    plt.setp(axes[1,0].get_xticklabels(), rotation=45)
    
    sns.countplot(x='residence', hue='stroke_event', data=df, ax=axes[1,1], palette='viridis')
    axes[1,1].set_title('Stroke by Residence Type')
    
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'categorical_stroke_analysis.png'))
    plt.close()

    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Features')
    plt.savefig(os.path.join(img_dir, 'correlation_heatmap.png'))
    plt.close()

    # Generate Statistical Summaries for the Markdown report
    stats = {
        "Total Patients": len(df),
        "Stroke Cases": int(df['stroke_event'].sum()),
        "Non-Stroke Cases": int(len(df) - df['stroke_event'].sum()),
        "Stroke Incidence Rate": f"{(df['stroke_event'].mean() * 100):.2f}%",
        "Average Age": f"{df['age'].mean():.2f}",
        "Average BMI": f"{df['bmi_value'].mean():.2f}",
        "Average Glucose": f"{df['glucose_level'].mean():.2f}"
    }
    
    # Age Group breakdown
    age_group_dist = df['age_group'].value_counts().to_dict()
    
    # Smoking Habits breakdown
    smoking_dist = df['smoking_habit'].value_counts().to_dict()

    print("STASTICAL_SUMMARY_START")
    print(stats)
    print("AGE_GROUP_DIST")
    print(age_group_dist)
    print("SMOKING_DIST")
    print(smoking_dist)
    print("STASTICAL_SUMMARY_END")

if __name__ == "__main__":
    generate_report_assets('cleaned_healthcare_data.csv')
