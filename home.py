import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("📊 KNN Interest Rate Prediction App")

uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์ CSV ของคุณ", type=['csv'])

if uploaded_file is not None:
    # โหลดข้อมูล
    df = pd.read_csv(uploaded_file)
    st.write("✅ ข้อมูลที่อัปโหลด:")
    st.dataframe(df.head())

    # ตรวจสอบว่ามีคอลัมน์ target ไหม
    if 'interest_rate' not in df.columns:
        st.error("⛔ ไม่พบคอลัมน์ 'interest_rate' ในไฟล์ CSV ของคุณ กรุณาตรวจสอบชื่อคอลัมน์")
    else:
        # เตรียมข้อมูล
        target_column = 'interest_rate'
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # จัดการข้อมูล
        X = pd.get_dummies(X)
        X.fillna(X.mean(), inplace=True)
        y.fillna(y.mean(), inplace=True)

        # แบ่งข้อมูล
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # สเกลข้อมูล
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # หาค่า K ที่ดีที่สุด
        k_range = range(1, 21)
        mse_scores = []

        for k in k_range:
            knn = KNeighborsRegressor(n_neighbors=k)
            scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
            mse_scores.append(-scores.mean())

        best_k = k_range[np.argmin(mse_scores)]
        st.success(f"✅ ค่า K ที่ดีที่สุดคือ: {best_k}")

        # สร้างโมเดลด้วย K ที่ดีที่สุด
        knn_best = KNeighborsRegressor(n_neighbors=best_k)
        knn_best.fit(X_train_scaled, y_train)
        y_pred = knn_best.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"📈 Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"📈 R-squared (R²): {r2:.4f}")

        # แบบฟอร์มทำนายจากข้อมูลใหม่
        st.subheader("🔮 ลองทำนายดอกเบี้ยจากข้อมูลที่คุณใส่")

        # ฟอร์มกรอกค่าจากผู้ใช้ (ตามฟีเจอร์ใน X)
        input_data = {}
        for col in X.columns:
            value = st.number_input(f"{col}", value=0.0)
            input_data[col] = value

        if st.button("ทำนาย"):
            new_df = pd.DataFrame([input_data])
            new_df = new_df.reindex(columns=X.columns, fill_value=0)  # เผื่อ user ไม่ใส่ครบ
            new_df_scaled = scaler.transform(new_df)
            prediction = knn_best.predict(new_df_scaled)
            st.success(f"🎯 ค่าดอกเบี้ยที่คาดการณ์ได้คือ: {prediction[0]:.2f}")
