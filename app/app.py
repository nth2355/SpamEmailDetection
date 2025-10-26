import pickle
from flask import Flask, request, render_template, jsonify
import os

MODEL_PATH = '../models/best_spam_detector.pkl'

app = Flask(__name__, template_folder='templates')
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    print(f"Đã tải mô hình thành công từ {MODEL_PATH}")
except FileNotFoundError:
    MODEL_PATH_ALT = 'models/best_spam_detector.pkl'
    try:
        with open(MODEL_PATH_ALT, 'rb') as file:
            model = pickle.load(file)
        print(f"Đã tải mô hình thành công từ {MODEL_PATH_ALT}")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file mô hình. Vui lòng kiểm tra {MODEL_PATH} hoặc {MODEL_PATH_ALT}")
        model = None
except Exception as e:
    print(f"LỖI KHÔNG TẢI ĐƯỢC MÔ HÌNH: {e}")
    model = None


@app.route('/')
def home():
    # Hiển thị trang chủ
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Lỗi: Mô hình chưa được tải."), 500

    message = request.form.get('message_text') 

    try:
        if not message:
            return render_template('index.html', prediction_text='Vui lòng nhập nội dung cần kiểm tra.', input_text=message)

        # 1. Lấy xác suất dự đoán (predict_proba)
        probabilities = model.predict_proba([message])[0] 
        
        # 2. Tính xác suất dưới dạng phần trăm
        prob_ham = probabilities[0] * 100  
        prob_spam = probabilities[1] * 100 
        
        # 3. Quyết định nhãn cuối cùng 
        if prob_spam >= 50:
            final_label = "🚨 THƯ RÁC (SPAM)"
        else:
            final_label = "✔️ THƯ HỢP LỆ (HAM)"
        
        # 4. Trả về kết quả (Dòng return thành công)
        return render_template('index.html', 
                                final_label=final_label,
                                prob_ham=f"{prob_ham:.2f}%",
                                prob_spam=f"{prob_spam:.2f}%",
                                input_text=message)
    
    except Exception as e:
        print(f"LỖI TRONG CHỨC NĂNG DỰ ĐOÁN: {e}")
        return render_template('index.html', 
                               prediction_text=f"Đã xảy ra lỗi hệ thống: {e}", 
                               input_text=message)
    

if __name__ == '__main__':
    # Chạy ứng dụng web
    app.run(debug=True)