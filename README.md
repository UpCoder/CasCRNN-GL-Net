### GL-BD-LSTM
##### Thanks for []
- Run dataset/medicalimage_to_tfrecords_original.py 将原始的mhd转化为tfrecord格式。
注意，我们对每个phase都有三个channel，R Bchannel代表的是病灶，Gchannel代表的是liver
- Run train.py 训练GL-BDLSTM， 此网络结构是承担对patch的分类
- Run evaluate_original.py 测试模型在验证集上的效果，注意，这里我们使用的original，也是直接从mhd文件中提取ROI和patch，不做任何截断操作
- Run generate_roi_feature)original.py 利用GL-BD-LSTM生成roi-level feature
- Run roi_fetaure_classifier.py 得到最后的roi级别的分类效果