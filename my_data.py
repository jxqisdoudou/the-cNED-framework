import pandas as pd
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
import numpy as np


def read_function_data(
        metadata_names=['WT', 'Longitude', 'Latitude', 'AT', 'Pre', 'Cond', 'pH', 'DO', 'Elevation',
                        'Cropland', 'Impervious', 'TN', 'TP', 'DOC', 'Chla', 'Distance'],
        random_state=42,
        function_filename='../Datasets/filtered_function.csv',
        metadata_filename='../Datasets/metadata.csv'):
    # 读取功能表和元数据
    function_data = pd.read_csv(function_filename, index_col=0, header=None, sep=',').T
    function_data = function_data.set_index('Functions')
    function_data = function_data.astype('float32')
    metadata = pd.read_csv(metadata_filename, sep=',')
    metadata = metadata.set_index('SAMPLE')

    # 提取所需的环境特征列
    domain = metadata[metadata_names]

    # 归一化处理（使用 MinMaxScaler）
    scaler_function = MinMaxScaler()
    scaler_domain = MinMaxScaler()

    function_normalized = pd.DataFrame(scaler_function.fit_transform(function_data),
                                       columns=function_data.columns,
                                       index=function_data.index)

    domain_normalized = pd.DataFrame(scaler_domain.fit_transform(domain),
                                     columns=domain.columns,
                                     index=domain.index)

    # 合并功能表和环境特征
    df = pd.concat([function_normalized, domain_normalized], axis=1, sort=True, join='outer')

    # 拆分训练集和测试集
    df_function_train, df_function_test, df_domain_train, df_domain_test = \
        train_test_split(df[function_data.columns], df[domain.columns], test_size=0.1,
                         random_state=random_state)

    # 打印训练集的前几行以检查数据
    print("Function training data:")
    print(df_function_train.head())
    print("\nDomain training data:")
    print(df_domain_train.head())

    return df_function_train, df_function_test, df_domain_train, df_domain_test, function_data.columns, domain.columns


def preprocess_metadata(metadata_filename, metadata_names):
    # 读取元数据
    metadata = pd.read_csv(metadata_filename, sep=',')
    metadata = metadata.set_index('SAMPLE')

    # 提取所需的环境特征列
    domain = metadata[metadata_names]

    # 编码时间（如果存在）
    if 'YEAR' in metadata_names:
        domain = pd.concat([domain, pd.get_dummies(domain['YEAR'], prefix='Year')], axis=1)
        domain = domain.drop(['YEAR'], axis=1)
    if 'MONTH' in metadata_names:
        domain = pd.concat([domain, pd.get_dummies(domain['MONTH'], prefix='Month')], axis=1)
        domain = domain.drop(['MONTH'], axis=1)

    # 归一化处理环境特征（使用 MinMaxScaler）
    scaler = MinMaxScaler()
    domain_normalized = pd.DataFrame(scaler.fit_transform(domain), columns=domain.columns, index=domain.index)

    return domain_normalized

# 示例调用
# df_function_train, df_function_test, df_domain_train, df_domain_test, function_columns, domain_columns = read_function_data()
