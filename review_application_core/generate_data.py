import os
from common.data_utils import sent_TokenizeFunct, jaccard, data_sampling, load_spark, load_data
import pyspark.sql.functions as f
from tqdm import tqdm


NOW_DIR = os.getcwd()
DATA_DR = os.path.join(NOW_DIR, 'data', 'yelp', 'small')
dataset_list = os.listdir(DATA_DR)

dataset_name_list = [[dataset_name.split('.')[0], dataset_name] for dataset_name in dataset_list]
dataset_dr_list = [{dataset_name: str(os.path.join(DATA_DR, dataset_path))}
                   for dataset_name, dataset_path in dataset_name_list]

sqlContext = load_spark("local", "advertising_detection_generate_data")
df_dict = {next(iter(data_info.keys())): load_data(sqlContext, next(iter(data_info.values())))
           for data_info in tqdm(dataset_dr_list)}

join_df = df_dict['yelp_academic_dataset_review'].join(df_dict['yelp_academic_dataset_business'], 'business_id',
                                                       how='left_outer')
join_df = join_df.join(df_dict['yelp_academic_dataset_user'], 'user_id', how='left_outer')
join_df = join_df.filter(join_df.text.isNotNull()).filter(join_df.categories.isNotNull()).filter(
    join_df.user_id.isNotNull()).filter(join_df.business_id.isNotNull()).filter(
    join_df.review_stars.isNotNull()).filter(join_df.business_stars.isNotNull()).filter(
    join_df.average_stars.isNotNull())

DataSamplingRDD = join_df.rdd.map(data_sampling).filter(bool)
sampling_df = sqlContext.createDataFrame(DataSamplingRDD)
sampling_df = sampling_df.select(f.col("_1").alias("text"), f.col("_2").alias("categories"), f.col("_3").alias("label"))
SAVE_DIR = os.path.join(NOW_DIR, 'data', 'False_Exaggerated_advertisement.csv')
sampling_df.write.csv(SAVE_DIR, header=True)
