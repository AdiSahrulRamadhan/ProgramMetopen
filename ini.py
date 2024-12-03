from sklearn.model_selection import train_test_split

# Pembagian data
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=123, stratify=y
)

# Output jumlah data
print(f"Data Training: {X_train.shape[0]}")
print(f"Data Testing: {X_test.shape[0]}")
