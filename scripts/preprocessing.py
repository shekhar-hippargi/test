import pandas as pd
import numpy as np
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns

def dataPreprocess(excelDataset):
    excel_data = pd.read_excel(excelDataset)
    # print(excel_data.head())
    excel_data.to_csv('../results/preprocessed/credit.csv', index=False)

    columns = ['Case_no', 'checking_status', 'credit_history', 'purpose', 'credit_amount', 'saving_status',
               'employment', 'personal_status', 'age', 'job', 'class']
    credit = pd.read_csv('../results/preprocessed/credit.csv', names=columns)

    # Pandas-Profiling to visualize the original dataset and explore insights of each variable
    report = credit.profile_report(title="Pandas profiling")
    report.to_file(output_file='../results/preprocessed/profile.html')

    # plt.boxplot(credit.age)
    # plt.show()
    # sns.distplot(credit.age)
    # plt.show()
    # sns.pairplot(credit)
    # plt.show()

    # Replacing values of age less than 6 and greater than 75 by mean of customers between age (6-75) in 'age' column
    mean = credit.loc[credit['age'].between(6, 76, inclusive=False), 'age'].mean()
    credit['age'] = np.where(credit['age'] > 75, mean, credit['age'])
    credit['age'] = np.where(credit['age'] < 6, mean, credit['age'])

    # Rectified spell errors in 'purpose' column
    credit.purpose.replace({'busines':'business', 'Eduction':'education', 'busness':'business','ather':'other',
                            'Radio/Tv':'radio/tv'}, inplace=True)

    '''
    credit.job.value_counts(): o/p
    skilled                        628
    'unskilled resident'           200
    'high qualif/self emp/mgmt'    148
    'unemp/unskilled non res'       22
    yes                              2
    '''
    #Replaced 'yes' to 'skilled' as majority were skilled and it won't as affect the result much as there were only 2 records
    credit.job.replace('yes', 'skilled', inplace=True)

    # Applying label encoding to categorical variables(columns)
    def labelEncoder(column):
        dict = {}
        unique_values = list(credit[column].unique())
        for i in range(len(unique_values)):
            dict[unique_values[i]] = i
        credit[column] = credit[column].map(dict)

    labelEncoder('checking_status')
    labelEncoder('credit_history')
    labelEncoder('purpose')
    labelEncoder('saving_status')
    labelEncoder('employment')
    labelEncoder('personal_status')
    labelEncoder('job')
    labelEncoder('class')

    # Generate preprocessed dataset
    credit.to_csv('../results/preprocessed/preprocessed.csv', index=False)

    # Post Label Encoding Pandas-Profiling
    report = credit.profile_report(title="Pandas profiling Preprocessed Dataset")
    report.to_file(output_file='../results/preprocessing_op/PreprocessedProfileReport.html')

dataPreprocess(r"../data/credit.xlsx")