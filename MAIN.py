import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.optimizers import Adam
from data_import import *
from train_2 import perform_experiment_2
from transfer_learning import train_tl_noEnsemble, test_model_tl_noEnsemble
from layers import Percentage
from loss import MakeLoss, LossBrayCurtis, LossMeanSquaredErrorWrapper, LossCategoricalCrossentropyWrapper
from metric import *
tf.compat.v1.enable_eager_execution()
print(tf.executing_eagerly())  # 返回 True 说明已经启用了
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. 读取数据集
df_microbioma_train, df_microbioma_test, df_domain_train, df_domain_test, otu_columns, domain_columns = read_data(
    metadata_names=['WT','Longitude','Latitude','AT','Pre','Cond','pH','DO','Elevation',
                        'Cropland','Impervious','TN','TP','DOC','Chla','Distance'],
    otu_filename='C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Datasets/Phylum_relative_abundance.csv',
    metadata_filename='C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Datasets/filled_metadata.csv',
)

data_microbioma_train = df_microbioma_train.to_numpy(dtype=np.float32)
data_microbioma_test = df_microbioma_test.to_numpy(dtype=np.float32)
data_domain_train = df_domain_train.to_numpy(dtype=np.float32)
data_domain_test = df_domain_test.to_numpy(dtype=np.float32)

# 记录训练时间
start_time = time.time()  # 记录开始时间

#2. 训练自动编码器模型
experiment_metrics, models, results = perform_experiment_2(
    cv_folds=0,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    optimizer=Adam,
    learning_rate_scheduler=None,
    input_transform=Percentage,
    output_transform=tf.keras.layers.Softmax,
    reconstruction_loss=MakeLoss('bray_curtis', Percentage().call, None),
    latent_space=10,
    layers=[128, 64, 32],
    activation='tanh',
    activation_latent='tanh',
    data_microbioma_train=data_microbioma_train,
    data_domain_train=None,
    show_results=True,
    device='/CPU:0'
)

# 保存编码器和解码器模型
model, encoder, _, decoder = models[0]

# 3. 构建潜在空间预测模型
def model_fn_latent():
    in_layer = layers.Input(shape=(data_domain_train.shape[1],))
    net = layers.Dense(128, activation='tanh')(in_layer)
    net = layers.Dense(64, activation='tanh')(net)
    net = layers.Dense(32, activation='tanh')(net)
    net = layers.Dense(16, activation='tanh')(net)
    out_layer = layers.Dense(latent_train.shape[1], activation=None)(net)
    model = keras.Model(inputs=[in_layer], outputs=[out_layer], name='model')
    model.compile(optimizer=Adam(lr=0.001), loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

latent_train = encoder.predict(data_microbioma_train)
result_latent, model_latent = train_tl_noEnsemble(model_fn_latent, latent_train, latent_train, data_domain_train, data_domain_train, epochs=100, batch_size=16, verbose=-1)

end_time = time.time()  # 记录结束时间
training_duration = end_time - start_time  # 计算训练耗时
print(f"Training time: {training_duration:.2f} seconds")  # 输出训练耗时

# 使用环境数据进行测试和预测
latent_test = model_latent.predict(data_domain_test)
predictions = decoder.predict(latent_test)

# 评估测试结果
metrics_results = test_model_tl_noEnsemble(model_latent, decoder, Percentage, tf.keras.layers.Softmax, otu_columns, data_microbioma_test, data_domain_test)

# 保存模型
with CustomObjectScope({'Percentage': Percentage}):
    encoder.save('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/Models/encoder_biome.h5')
    decoder.save('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/Models/decoder.h5')
    model_latent.save('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/Models/encoder_domain_model_latent.h5')

encoder_biome = encoder
encoder_domain = model_latent

# 保存数据
np.save('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/data_microbioma_train.npy', data_microbioma_train)
np.save('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/data_microbioma_test.npy', data_microbioma_test)
np.save('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/data_domain_train.npy', data_domain_train)
np.save('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/data_domain_test.npy', data_domain_test)
np.save('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/latent_train.npy', latent_train)
np.save('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/latent_test.npy', latent_test)
np.save('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/otu_columns.npy', otu_columns)
np.save('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/domain_columns.npy', domain_columns)
np.save('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/predictions.npy', predictions)
# 保存预测结果
def save_predicted_otu_table_and_latent(pred, pred_latent, sample_names, otu_names, suffix=''):
    df_otu = pd.DataFrame(pred, index=sample_names, columns=otu_names)
    df_otu.T.to_csv('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/otus_' + suffix + '.csv', index=True, header=True, sep=',')

    df_latent = pd.DataFrame(pred_latent, index=sample_names)
    df_latent.T.to_csv('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/latent_' + suffix + '.csv', index=True, sep=',')

    return df_otu, df_latent

# 预测并保存结果
pred_latent_biome = encoder_biome.predict(data_microbioma_test)
pred_biome = decoder.predict(pred_latent_biome)
_, _ = save_predicted_otu_table_and_latent(pred_biome, pred_latent_biome, df_microbioma_test.index, df_microbioma_test.columns, 'reconstAEfromBiome')

pred_latent = encoder_domain.predict(data_domain_test)
pred_domain = decoder.predict(pred_latent)
df_pred_otu, df_pred_latent = save_predicted_otu_table_and_latent(pred_domain, pred_latent, df_microbioma_test.index, df_microbioma_test.columns, 'predFromDomain')

df_microbioma_test.to_csv('C:/Users/22152/Desktop/python/DeepLatentMicrobiome-test/Results/otu_original.csv', index=True)



