API CheatSheet
==============

The following APIs are applicable for all detector models for easy use.

* :func:`deepod.core.base_model.BaseDeepAD.fit`: Fit detector. y is ignored in unsupervised methods.
拟合检测器
* :func:`deepod.core.base_model.BaseDeepAD.decision_function`: Predict raw anomaly score of X using the fitted detector.
使用拟合检测器预测 X 的原始异常分数
* :func:`deepod.core.base_model.BaseDeepAD.predict`: Predict if a particular sample is an outlier or not using the fitted detector.
使用拟合好的的检测器预测特定样本是否为异常值


Key Attributes of a fitted model:

* :attr:`deepod.core.base_model.BaseDeepAD.decision_scores_`: The outlier scores of the training data. The higher, the more abnormal.
  Outliers tend to have higher scores.
训练数据的异常值分数：越高越不正常   异常点倾向于有更高的分数
* :attr:`deepod.core.base_model.BaseDeepAD.labels_`: The binary labels of the training data. 0 stands for inliers and 1 for outliers/anomalies.
训练数据的二进制标签：0 代表正常值，1 代表异常值/异常值

See base class definition below:

deepod.core.base_model module
-----------------------

.. automodule:: deepod.core.base_model
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

