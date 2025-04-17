import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------- STEP 1: โหลดข้อมูล ----------
df = pd.read_csv('cirrhosis.csv')  # แก้ชื่อไฟล์ตรงนี้
print("ข้อมูลตัวอย่าง:")
print(df.head())

# ---------- STEP 2: เตรียมข้อมูล ----------
# แก้ 'interest_rate' ให้ตรงกับชื่อคอลัมน์ที่คุณจะทำนาย
target_column = 'interest_rate'
X = df.drop(target_column, axis=1)
y = df[target_column]

# แปลงข้อมูล category เป็นตัวเลข (One-Hot Encoding)
X = pd.get_dummies(X)

# แบ่งข้อมูลเป็น train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# สเกลข้อมูล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- STEP 3: หาค่า K ที่ดีที่สุด ----------
k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

best_k = k_range[np.argmin(cv_scores)]
print(f"\n✅ ค่า K ที่ดีที่สุดคือ: {best_k}")
print(f"   ด้วยค่า MSE ต่ำสุด: {min(cv_scores):.4f}")

# ---------- STEP 4: สร้างโมเดลด้วยค่า K ที่ดีที่สุด ----------
knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)

# ---------- STEP 5: ทำนาย ----------
y_pred = knn_best.predict(X_test_scaled)

# ---------- STEP 6: สรุปผล ----------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 สรุปผล:")
print(f"🔹 Mean Squared Error (MSE): {mse:.4f}")
print(f"🔹 R-squared (R²): {r2:.4f}")

# ---------- STEP 7: วาดกราฟ ----------
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, marker='o', linestyle='--', color='blue')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K = {best_k}')
plt.title('K vs. Mean Squared Error (Cross Validation)')
plt.xlabel('Number of Neighbors: K')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# กราฟเปรียบเทียบค่าทำนายกับค่าจริง
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.title('Actual vs Predicted Interest Rates')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()
