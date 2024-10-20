import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from configs import  KERATIN,IMMUNE_PROTEINS,PROTEOMIC_DATA_DISCOVERY,PROTEOMIC_DATA_REPLICATION,CLINICAL_DATA_DISCOVERY,CLINICAL_DATA_REP,DELTA_ALSFRS_REP,DELTA_ALSFRS_DIS


# ---------------------------------load data-------------------------------------------
#proteomic data
def load_raw_proteomics_discovery():
    df = pd.read_csv(PROTEOMIC_DATA_DISCOVERY)
    df = df.set_index('sample number')
    print(f"Raw data shape: {df.shape}")
    return df


def load_raw_proteomics_replication():
        df = pd.read_csv(PROTEOMIC_DATA_REPLICATION)
        df = df.set_index('sample number')
        print(f"Raw data shape: {df.shape}")
        return df
 
#proteomic data prepared to multicalss
def load_replication_data_for_classification():
    # Load the delta ALSFRS data
    df = pd.read_csv(DELTA_ALSFRS_REP)
    mean_month_from_onset = df['month_from_onset'].mean()
    print(f"Average monthly time from diagnosis to blood test: {mean_month_from_onset}")
    
    # Count the number of rows of 'ALSFRS_slope (-unit/month)'
    slow = (df['ALSFRS_slope (-unit/month)'] < 0.5).sum()
    fast = (df['ALSFRS_slope (-unit/month)'] > 1.5).sum()
    intermediate = ((df['ALSFRS_slope (-unit/month)'] >= 0.5) & 
               (df['ALSFRS_slope (-unit/month)'] <= 1.5)).sum()

    print(f'Slow: {slow}')
    print(f'Fast: {fast}')
    print(f'Between: {intermediate}')
    
    # Add the classification column 
    def classify_slope(slope):
        if slope < 0.5:
            return 0
        elif 0.5 <= slope <= 1.5:
            return 1
        else:
            return 2

    df['Target'] = df['ALSFRS_slope (-unit/month)'].apply(classify_slope)
    
    # Drop unnecessary columns
    df = df.drop(columns=['ID', 'month_from_onset', 'ALSFRS_slope (-unit/month)'])
    
    # Load the proteomic_rep data
    proteomic_rep = pd.read_csv(PROTEOMIC_DATA_REPLICATION)
    
    # Merge the delta ALSFRS data with proteomic_rep
    df = pd.merge(df, proteomic_rep, on='sample number', how='left')
    df = df.set_index("sample number")
    
    return df

def load_discovery_data_for_classification():
    
    df = pd.read_csv(DELTA_ALSFRS_DIS)
    
    mean_month_from_onset = df['month_from_onset'].mean()
    print(f"Average monthly time from diagnosis to blood test: {mean_month_from_onset}")
    
    # Count the number of 'ALSFRS_slope (-unit/month)'
    slow = (df['deltaFRS (-unit/months)'] < 0.5).sum()
    fast = (df['deltaFRS (-unit/months)'] > 1.5).sum()
    intermediate = ((df['deltaFRS (-unit/months)'] >= 0.5) & 
               (df['deltaFRS (-unit/months)'] <= 1.5)).sum()

    
    print(f'Slow(class 0): {slow}')
    print(f'Fast(class 1): {fast}')
    print(f'intermediate (class 2): {intermediate}')
    
    # Add the classification column 
    def classify_slope(slope):
        if slope < 0.5:
            return 0
        elif 0.5 <= slope <= 1.5:
            return 1
        else:
            return 2

    df['Target'] = df['deltaFRS (-unit/months)'].apply(classify_slope)
    
    # Drop unnecessary columns
    df = df.drop(columns=['ID', 'month_from_onset', 'deltaFRS (-unit/months)'])
    
    # Load the proteomic_dis
    proteomic_dis = pd.read_csv('discovery_processed.csv')
    
    # Merge the delta ALSFRS data with proteomic_dis
    df = pd.merge(df, proteomic_dis, on='sample number', how='left')
    
    df = df.set_index("sample number")
    
    return df


# ----------------------------------------------------------------------------------------

#plots thast describe the data
def describe_proteins_and_samples(df):
   
    # Use describe() to calculate descriptive statistics for proteins (columns)
    # protein_stats = df.describe()
    # print(protein_stats)
    
    # Extract mean and variance for histogram plotting
    protein_means = df.mean()
    protein_variances = df.var()
    

    # Plot the distribution of protein means
    plt.figure(figsize=(8, 4))
    plt.hist(protein_means, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Protein Means')
    plt.xlabel('Mean Protein Abundance')
    plt.ylabel('Frequency')
    plt.show()

    # Plot the distribution of protein variances
    plt.figure(figsize=(8, 4))
    plt.hist(protein_variances, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Protein Variances')
    plt.xlabel('Variance of Protein Abundance')
    plt.ylabel('Frequency')
    plt.show()

    # Calculate statistics for samples (rows)
    sample_protein_counts = df.count(axis=1)

    # Plot the distribution of protein counts per sample
    plt.figure(figsize=(8, 4))
    plt.hist(sample_protein_counts, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Protein Counts per Sample')
    plt.xlabel('Number of Detected Proteins per Sample')
    plt.ylabel('Frequency of Samples')
    plt.show()
    
    
    return df

def describe_clinical_data(df):
     # Plotting the box plot for "Survival_from_onset (months)"
    plt.figure(figsize=(8, 4))
    plt.boxplot(df["Survival_from_onset (months)"])
    plt.title("Box Plot of Survival from Onset (months)")
    plt.ylabel("Survival from Onset (months)")
    plt.grid(True)
    
    plt.figure(figsize=(8, 4))
    plt.hist(df["Survival_from_onset (months)"], bins=10, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Survival time from Onset')
    plt.xlabel('Survival time(month)')
    plt.ylabel('Frequency of Samples')
    plt.show()
    
    plt.figure(figsize=(8, 4))
    df.boxplot(column="Survival_from_onset (months)", by="Sex", grid=False)
    num_males = df[df['Sex'] == 0].shape[0]
    num_females = df[df['Sex'] == 1].shape[0]
    plt.text(1, df["Survival_from_onset (months)"].max(), f'N = {num_males}', horizontalalignment='center', verticalalignment='center')
    plt.text(2, df["Survival_from_onset (months)"].max(), f'N = {num_females}', horizontalalignment='center', verticalalignment='center')
    
    plt.title("Box Plot of Survival from Onset (months) by Sex")
    plt.suptitle("")
    plt.xlabel("Sex")
    plt.ylabel("Survival from Onset (months)")
    plt.xticks([1, 2], ["Male", "Female"])  # Set custom x-tick labels
    plt.grid(True)
    plt.show()
    
    # Histogram for "Age Onset (years)" with average line
    plt.figure(figsize=(8, 4))
    plt.hist(df["Age Onset (years)"], bins=10, edgecolor='black', alpha=0.7)
    average_age_onset = df["Age Onset (years)"].mean()
    plt.axvline(average_age_onset, color='red', linestyle='dotted', linewidth=2)
    #plt.text(average_age_onset, plt.ylim()[1] * 0.9, f'Average: {average_age_onset:.2f} years', color='red', horizontalalignment='right')
    plt.title('Distribution of Age Onset (years)')
    plt.xlabel('Age Onset (years)')
    plt.ylabel('Frequency of Samples')
    plt.show()
    
    return df
# ----------------------------------------------------------------------------------------
#function to drop IMMUNE_PROTEINS and KERATIN proteins and proteins with low variance 

def drop_protein(df):
    """
   
    Parameters:
    IMMUNE_PROTEINS (list): List of immune protein columns to drop.
    KERATIN (list): List of keratin columns to drop.
    threshold (float): Variance threshold for filtering columns.
    """
    # Drop immune protein columns and print the shape
    #df = df.set_index('sample_ID')
    df = df.drop(columns=IMMUNE_PROTEINS)
    print(f"Shape after dropping immune proteins: {df.shape}")

    # Drop keratin columns and print the shape
    df = df.drop(columns=KERATIN)
    print(f"Shape after dropping keratin: {df.shape}")

    # Filter out proteins that are found in less than 50% of the tested samples
    required_count = len(df) * 0.5
    df = df.loc[:, df.count() >= required_count]
    print(f"Shape after filtering proteins >50%: {df.shape}")
    return df

def low_variance_proteins(df, threshold=0.5) :
    # Calculate variance and apply variance threshold filtering - Remove proteins with variance lower than threshold
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    df = df[df.columns[selector.get_support(indices=True)]]
    print(f"Proteins with variance above threshold (threshold={threshold}): {df.shape}")
    
    # Plot the distribution of variances with a threshold line
    variance = df.var()
    print(len(variance))
    plt.figure(figsize=(8, 4))
    plt.hist(variance, bins=40, alpha=0.7, edgecolor='black')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    plt.title('Distribution of Protein Variances with Threshold Line')
    plt.xlabel('Variance of Protein Abundance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    return df

# --------------------------------------------------------------------------------
#L1 normalisation and log2 transformation
def norm_log(df):
    # Normalize 1
    row_sums = df.abs().sum(axis=1)
    normalized_df = df.div(row_sums, axis=0)

    # Check that the sum of each row is 1
    row_sums_check = normalized_df.sum(axis=1)
    print(f"Sum of rows is 1: {np.allclose(row_sums_check, 1)}")
    print(f"Sum of the first 3 rows after normalization: {row_sums_check.head(3)}")

    # Apply log2 transformation, adding a small value to avoid log(0)
    normalized_df = normalized_df.astype(float)
    normalized_df = np.log2(normalized_df + 1e-10)  # Optional: Apply log2 transformation
    

    return normalized_df

# ----------------------------------------------------------------------------
#Merg protein data with clinical data.

def merge_proteins_and_clinical_data_discovery(df):
    clinical_data = pd.read_csv(CLINICAL_DATA_DISCOVERY)
    print(f"Loading clinical data from {CLINICAL_DATA_DISCOVERY}\nShape: {clinical_data.shape}")
    
    clinical_data1 = clinical_data[["sample number", "ALSFRS score (unit)","Age Onset (years)",'Sex','Disease Format']]
    clinical_data1.set_index("sample number", inplace=True)
    df_with_clinical = df.join(clinical_data1, how='inner')
    # scaler = StandardScaler()
    # df_with_clinical = pd.DataFrame(scaler.fit_transform(df_with_clinical), 
    #                                columns=df_with_clinical.columns, 
    #                                index=df_with_clinical.index)
    clinical_data2 = clinical_data[["sample number","Survival_from_onset (months)","Status dead=1"]]
    clinical_data2.set_index("sample number", inplace=True)
    df_with_clinical = df_with_clinical.join(clinical_data2, how='inner')
    
    sex_mapping = {'M': 0, 'F': 1}
    df_with_clinical['Sex'] = df_with_clinical['Sex'].map(sex_mapping)
    disease_format_mapping = {'Limb': 0, 'Bulbar': 1}
    df_with_clinical['Disease Format'] = df_with_clinical['Disease Format'].map(disease_format_mapping)
    return df_with_clinical


def merge_proteins_and_clinical_data_rep(df):
    clinical_data = pd.read_csv(CLINICAL_DATA_REP)
    print(f"Loading clinical data from {CLINICAL_DATA_REP}\nShape: {clinical_data.shape}")
    
    clinical_data1 = clinical_data[["sample number", "ALSFRS score (unit)","Age Onset (years)",'Sex','Disease Format']]
    clinical_data1.set_index("sample number", inplace=True)
    df_with_clinical = df.join(clinical_data1, how='inner')
    # scaler = StandardScaler()
    # df_with_clinical = pd.DataFrame(scaler.fit_transform(df_with_clinical), 
    #                                columns=df_with_clinical.columns, 
    #                                index=df_with_clinical.index)
    clinical_data2 = clinical_data[["sample number","Survival_from_onset (months)","Status dead=1"]]
    clinical_data2.set_index("sample number", inplace=True)
    df_with_clinical = df_with_clinical.join(clinical_data2, how='inner')
    
    sex_mapping = {'M': 0, 'F': 1}
    df_with_clinical['Sex'] = df_with_clinical['Sex'].map(sex_mapping)
    disease_format_mapping = {'Limb': 0, 'Bulbar': 1}
    df_with_clinical['Disease Format'] = df_with_clinical['Disease Format'].map(disease_format_mapping)
    return df_with_clinical
