import unittest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from price_analysis import (
    predict_property_price,
    find_comparable_properties,
    generate_price_trends,
    calculate_price_range,
    get_neighborhood_stats,
    calculate_confidence_interval,
    get_model_explanation
)
 
class TestPriceAnalysis(unittest.TestCase):
    
    def setUp(self):
        """测试前的准备工作"""
        # 从CSV文件加载真实数据
        try:
            # 尝试多个可能的CSV路径
            csv_paths = [
                '../resources/house_samples_features.csv',
                'resources/house_samples_features.csv',
                './resources/house_samples_features.csv',
                '../../resources/house_samples_features.csv',
                '../house_samples_features.csv',
                './house_samples_features.csv'
            ]
            
            self.real_data_df = None
            for path in csv_paths:
                try:
                    if Path(path).exists():
                        self.real_data_df = pd.read_csv(path)
                        print(f"成功加载样本数据文件: {path}")
                        break
                except Exception as e:
                    print(f"尝试加载CSV {path} 失败: {e}")
            
            if self.real_data_df is None:
                raise FileNotFoundError("无法找到数据文件")
                
            # 随机抽取5行作为测试样本
            self.sample_df = self.real_data_df.sample(5, random_state=42)
            print(f"成功加载样本数据，共{len(self.sample_df)}行")
            
            # 展示样本基本信息
            print("\n样本数据基本信息:")
            for i, row in self.sample_df.iterrows():
                print(f"样本 {i}: 地址 '{row['std_address']}', 价格 ¥{row['y_label']:.2f}, "
                     f"类型 {row.get('prop_type', 'N/A')}, 卧室 {row.get('prop_bed', 'N/A')}, "
                     f"浴室 {row.get('prop_bath', 'N/A')}")
        except Exception as e:
            print(f"加载CSV文件失败: {e}") 
            raise e
        
        print(f"样本数据: {self.sample_df}")
            
        # 特征列列表 - 只选择数值型特征
        self.feature_cols = [col for col in self.real_data_df.columns 
                            if col not in ['y_label', 'std_address', 'prop_id'] 
                            ]
        
        #print(f"使用的特征列: {self.feature_cols}")
        
        # 加载真实的XGBoost模型
        try:
            # 尝试不同的模型路径
            model_paths = [
                '../model/xgb_model.joblib',
                'model/xgb_model.joblib',
                './model/xgb_model.joblib',
                '../../model/xgb_model.joblib'
            ]
            
            self.real_model = None
            for path in model_paths:
                try:
                    if Path(path).exists():
                        self.real_model = joblib.load(path)
                        print(f"成功加载真实XGBoost模型: {path}")
                        break
                except Exception as e:
                    print(f"尝试加载模型 {path} 失败: {e}")
            
            if self.real_model is None:
                print("警告: 无法加载真实模型，将使用Mock模型作为备用")
                self.real_model = MockModel(return_value=500000)
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.real_model = MockModel(return_value=500000)
        
    def test_predict_property_price(self):
        """测试房价预测功能"""
        # 使用真实数据中的第一个样本
        sample_row = self.sample_df.iloc[0]
        
        # 确保特征数据类型正确
        numeric_feature_cols = [col for col in self.feature_cols 
                               if pd.api.types.is_numeric_dtype(self.sample_df[col])]
        
        print(f"用于预测的特征列: {numeric_feature_cols[:9]}, 特征列数量: {len(numeric_feature_cols)}")
        
        # 预测价格
        pred_price, importance = predict_property_price(
            row=sample_row,
            model=self.real_model,
            feature_cols=numeric_feature_cols,  # 使用所有数值特征
            properties_df=self.real_data_df
        )
        
        # 验证预测价格是一个有效数字
        self.assertIsInstance(pred_price, float)
        self.assertGreater(pred_price, 0)
        
        # 验证特征重要性列表
        self.assertIsInstance(importance, list)
        self.assertTrue(all(isinstance(item, dict) for item in importance))
         
            
        print(f"预测价格: ¥{pred_price:.2f}，特征重要性数量: {len(importance)}")
        print("房产信息: 地址 '{}', 实际价格: ¥{:.2f}".format(
            sample_row['std_address'], sample_row['y_label']))
        print("预测误差: ¥{:.2f}".format(abs(pred_price - sample_row['y_label'])),"预测误差率: {:.2%}".format(abs(pred_price - sample_row['y_label'])/sample_row['y_label']))
        print("前三个重要特征:")
        print("feature_importance:",importance)
        for i, feat in enumerate(importance[:3]):
            print(f"  {i+1}. {feat['feature']}: 重要性 {feat['importance']:.4f}, 影响 {feat['effect']}")
        
    def test_find_comparable_properties(self):
        """测试查找可比房产功能"""
        # 使用真实数据中的样本
        sample_row = self.sample_df.iloc[0]
        prop_id = str(sample_row['prop_id'])
        
        # 寻找可比房产
        comparable_properties = find_comparable_properties(
            row=sample_row,
            prop_id=prop_id,
            properties_df=self.real_data_df
        )
        
        # 验证返回结果
        self.assertIsInstance(comparable_properties, list)
        print(f"可比房产数量: {len(comparable_properties)}")
        
        # 如果有可比房产，验证其结构
        if comparable_properties:
            comp = comparable_properties[0]
            self.assertIn('id', comp)
            self.assertIn('address', comp)
            self.assertIn('price', comp)
            self.assertIn('similarity', comp)
            
            # 打印详细信息
            print(f"当前房产: 地址 '{sample_row['std_address']}', ID {prop_id}, 价格 ¥{sample_row['y_label']:.2f}")
            print("找到的可比房产:")
            for i, comp in enumerate(comparable_properties):
                print(f"  {i+1}. 地址: '{comp['address']}', ID: {comp['id']}, 价格: ¥{comp['price']:.2f}, 相似度: {comp['similarity']}")
        
    def test_generate_price_trends(self):
        """测试生成价格趋势功能"""
        # 使用样本数据的实际价格
        sample_price = float(self.sample_df.iloc[0]['y_label'])
        
        # 生成价格趋势
        trends = generate_price_trends(sample_price)
        
        # 验证返回结果
        self.assertIsInstance(trends, list)
        self.assertEqual(len(trends), 12)  # 应该有12个月的数据
        
        # 验证趋势数据结构
        for trend in trends:
            self.assertIn('date', trend)
            self.assertIn('price', trend)
            self.assertIsInstance(trend['price'], float)
            
        print(f"生成价格趋势数据点: {len(trends)}，首月价格: ¥{trends[0]['price']:.2f}")
        
    def test_calculate_price_range(self):
        """测试计算价格区间功能"""
        # 使用样本数据的实际价格
        sample_price = float(self.sample_df.iloc[0]['y_label'])
        
        # 计算价格区间
        price_range = calculate_price_range(sample_price)
        
        # 验证返回结果
        self.assertIsInstance(price_range, dict)
        self.assertIn('min', price_range)
        self.assertIn('max', price_range)
        self.assertIn('most_likely', price_range)
        
        # 验证价格区间逻辑
        self.assertLess(price_range['min'], price_range['most_likely'])
        self.assertGreater(price_range['max'], price_range['most_likely'])
        
        print(f"价格区间: ¥{price_range['min']:.2f} - ¥{price_range['max']:.2f}，最可能价格: ¥{price_range['most_likely']:.2f}")
        
    def test_get_neighborhood_stats(self):
        """测试获取周边区域统计功能"""
        # 使用样本数据的实际价格
        sample_price = float(self.sample_df.iloc[0]['y_label'])
        
        # 使用样本的面积，如果存在的话
        sample_area = 0
        for area_col in ['internal_area']:
            if area_col in self.sample_df.columns:
                sample_area = float(self.sample_df.iloc[0].get(area_col, 0))
                if sample_area > 0:
                    break
                
        if sample_area == 0:
            sample_area = 100  # 默认值
        
        # 获取周边区域统计
        stats = get_neighborhood_stats(sample_price, sample_area)
        
        # 验证返回结果
        self.assertIsInstance(stats, dict)
        self.assertIn('avg_price', stats)
        self.assertIn('min_price', stats)
        self.assertIn('max_price', stats)
        self.assertIn('num_properties', stats)
        self.assertIn('price_trend', stats)
        
        print(f"周边区域统计: 平均价格¥{stats['avg_price']:.2f}，每平方米单价: ¥{stats['avg_price_per_sqm']:.2f}")
        
    def test_calculate_confidence_interval(self):
        """测试计算置信区间功能"""
        # 使用样本数据的实际价格
        sample_price = float(self.sample_df.iloc[0]['y_label'])
        
        # 计算置信区间
        ci = calculate_confidence_interval(sample_price)
        
        # 验证返回结果
        self.assertIsInstance(ci, dict)
        self.assertIn('lower_bound', ci)
        self.assertIn('upper_bound', ci)
        self.assertIn('confidence_level', ci)
        
        # 验证置信区间逻辑
        self.assertLess(ci['lower_bound'], ci['upper_bound'])
        
        print(f"置信区间: ¥{ci['lower_bound']:.2f} - ¥{ci['upper_bound']:.2f}，置信水平: {ci['confidence_level']}")
        
    def test_get_model_explanation(self):
        """测试获取模型解释功能"""
        # 使用样本数据的实际价格和特征重要性
        sample_row = self.sample_df.iloc[0]
        sample_price = float(sample_row['y_label'])
        
        # 确保特征数据类型正确
        numeric_feature_cols = [col for col in self.feature_cols 
                               if pd.api.types.is_numeric_dtype(self.sample_df[col])]
        
        # 使用predict_property_price获取实际的特征重要性
        _, feature_importance = predict_property_price(
            row=sample_row,
            model=self.real_model,
            feature_cols=numeric_feature_cols[:9],  # 使用前9个特征，与模型feature_importances_维度匹配
            properties_df=self.real_data_df
        )
        
        # 获取模型解释
        explanation = get_model_explanation(
            pred_price=sample_price,
            feature_importance=feature_importance,
            feature_cols=numeric_feature_cols[:9]
        )
        
        # 验证返回结果
        self.assertIsInstance(explanation, dict)
        self.assertIn('model_type', explanation)
        self.assertIn('r2_score', explanation)
        self.assertIn('mae', explanation)
        self.assertIn('top_positive_features', explanation)
        self.assertIn('top_negative_features', explanation)
        
        print(f"模型解释: R²: {explanation['r2_score']}, MAE: ¥{explanation['mae']:.2f}")
        print(f"主要正向特征: {explanation['top_positive_features']}")
        print(f"主要负向特征: {explanation['top_negative_features']}")


if __name__ == '__main__':
    #unittest.main() 
    #单个测试
    suite = unittest.TestSuite()
    #测试可比房产
    suite.addTest(TestPriceAnalysis('test_predict_property_price'))
    unittest.TextTestRunner().run(suite)
    #unittest.main(defaultTest='backend.test_price_analysis.TestPriceAnalysis.test_predict_property_price')