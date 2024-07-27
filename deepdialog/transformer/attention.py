import pandas as pd
from transformers import pipeline

# 感情分析のパイプラインをロード
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Excelファイルを読み込む
file_path = 'path_to_your_file/Data_Final_Report_fall_semester.xlsx'
df = pd.read_excel(file_path)

# 文章が含まれている列を選択 (仮に最初の列が文章だとする)
texts = df.iloc[:, 0].tolist()

# 感情分析を実行
results = [sentiment_pipeline(text) for text in texts]

# スコアを抽出
pos_scores = [result[0]['score'] for result in results if result[0]['label'] == 'POSITIVE']
neg_scores = [result[0]['score'] for result in results if result[0]['label'] == 'NEGATIVE']

# 最大スコアを見つける
max_pos_score = max(pos_scores) if pos_scores else None
max_neg_score = max(neg_scores) if neg_scores else None

# 件数をカウント
pos_count_0_to_05 = sum(0 <= score <= 0.5 for score in pos_scores)
pos_count_05_to_1 = sum(0.5 < score <= 1 for score in pos_scores)
neg_count_0_to_05 = sum(0 <= score <= 0.5 for score in neg_scores)
neg_count_05_to_1 = sum(0.5 < score <= 1 for score in neg_scores)

results_df = pd.DataFrame({
    'text': texts,
    'sentiment': [result[0]['label'] for result in results],
    'score': [result[0]['score'] for result in results]
})

# 結果をCSVファイルとして保存
results_df.to_csv('sentiment_analysis_results.csv', index=False)

max_pos_score, max_neg_score, pos_count_0_to_05, pos_count_05_to_1, neg_count_0_to_05, neg_count_05_to_1
