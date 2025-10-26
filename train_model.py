import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os
import pickle




DATA_PATH = 'data/spam.csv'
MODEL_PATH = 'models/best_spam_detector.pkl'
NLTK_STOPWORDS = stopwords.words('english')


def clean_text(text):
    text = str(text).lower() 
    
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    tokens = text.split()
    tokens = [word for word in tokens if word not in NLTK_STOPWORDS and word.isalpha()]
    return " ".join(tokens)


def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary')
    rec = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n--- Kết quả Đánh giá: {model_name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)
    
    # Trực quan hóa Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['HAM (0)', 'SPAM (1)'], 
                yticklabels=['HAM (0)', 'SPAM (1)'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return {'model': model_name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def train_and_compare_traditional():
    print("--- BẮT ĐẦU CHƯƠNG TRÌNH ---")
    
    # 1. Tải dữ liệu
    try:
            df = pd.read_csv(DATA_PATH) 
  
            if 'label' not in df.columns or 'text' not in df.columns:
                raise ValueError("File CSV phải có cột 'label' và 'text'.")

            print(f"Đã tải {len(df)} mẫu dữ liệu. Kiểm tra NA...") 
        
    except FileNotFoundError:
            print(f"LỖI QUAN TRỌNG: KHÔNG TÌM THẤY FILE DỮ LIỆU TẠI {DATA_PATH}.")
            return
    except Exception as e:
            print(f"LỖI KHÁC KHI TẢI DỮ LIỆU: {e}")
            return
    
    #2. Tiền xử lý dữ liệu
    print(f"Tổng số mẫu ban đầu: {len(df)}")
    #Loại bỏ các hàng có giá trị thiếu trong cột 'text' hoặc 'label'
    df.dropna(subset=['text', 'label'], inplace=True)
    
    df['label'] = df['label'].astype(int)
    
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    print(f"Tổng số mẫu sau khi loại bỏ NA: {len(df)}")
    print(f"Số mẫu label 1 (SPAM): {df['label'].sum()}")
    print(f"Số mẫu label 0 (HAM): {len(df) - df['label'].sum()}")
    
    
    #3. Phân chia tập dữ liệu
    X = df['cleaned_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    #4. Huấn luyện và đánh giá mô hình
    
    tfidf = TfidfVectorizer(max_features=5000)
    
    pipelines = {
        'Naive Bayes (NB)': Pipeline([('tfidf', tfidf), ('nb', MultinomialNB())]),
        'Logistic Regression (LR)': Pipeline([('tfidf', tfidf), ('lr', LogisticRegression(max_iter=200))])
    }
    
    results = []
    
    for name, pipeline in pipelines.items():
            print(f"Đang huấn luyện: {name}...")
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            results.append(evaluate_model(y_test, y_pred, name))
            
    #5. Lưu mô hình tốt nhất(dựa trên F1-Score)
    best_model = max(results, key=lambda x: x['f1'])
    best_pipeline = pipelines[best_model['model']]
    
    print("\n" + "=" * 50)
    print(f"🏆 MÔ HÌNH TỐT NHẤT TRUYỀN THỐNG: {best_model['model']}")
    print(f"Với F1-Score: {best_model['f1']:.4f}")
    print("=" * 50)
    
    # 6. Lưu Mô hình Tốt nhất
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(best_pipeline, file)
    print(f"\n Mô hình tốt nhất đã được lưu tại: {MODEL_PATH}")
    
    return results
if __name__ == "__main__":
    traditional_results = train_and_compare_traditional()