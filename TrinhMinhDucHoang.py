import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu mẫu: y gần đúng bằng 3x + 2
# Dùng x trong khoảng 0 đến 1 để loss giảm mượt và dễ quan sát hơn.
x = np.linspace(0, 1, 20)
y = 3 * x + 2

# Khởi tạo trọng số
w = 0.0
b = 0.0
learning_rate = 0.1
epochs = 80
loss_history = []

def predict(x, w, b):
    return w * x + b

def mse_loss(y_true, y_pred):
    error = y_pred - y_true
    return np.mean(error ** 2)

def compute_gradients(x, y_true, y_pred):
    error = y_pred - y_true
    grad_w = 2 * np.mean(x * error)
    grad_b = 2 * np.mean(error)
    return grad_w, grad_b

for epoch in range(epochs):
    # Tính giá trị dự đoán
    y_pred = predict(x, w, b)

    # Tính loss
    loss = mse_loss(y, y_pred)
    loss_history.append(loss)

    # Tính gradient
    grad_w, grad_b = compute_gradients(x, y, y_pred)

    # Cập nhật trọng số
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:3d} | loss = {loss:8.4f} | w = {w:6.3f} | b = {b:6.3f}")

def plot_training_result(loss_history, x, y_true, w, b):
    epochs_axis = np.arange(len(loss_history))
    y_pred = predict(x, w, b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_axis, loss_history, color="tab:blue", linewidth=2)
    axes[0].set_title("Loss theo epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss MSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(x, y_true, color="tab:orange", label="Dữ liệu thật")
    axes[1].plot(x, y_pred, color="tab:blue", linewidth=2, label="Đường dự đoán")
    axes[1].set_title(f"Đường thẳng học được: y = {w:.2f}x + {b:.2f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

plot_training_result(loss_history, x, y, w, b)

print("\nKết quả cuối cùng:")
print(f"w = {w:.4f}")
print(f"b = {b:.4f}")
print(f"Dự đoán khi x = 1.5: {predict(1.5, w, b):.4f}")

# Dữ liệu linear regression
x_linear = np.array([-3.0, -2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4, 3.0])
y_linear = np.array([-8.7, -6.6, -5.4, -4.0, -2.4, -1.0, 0.0, 1.6, 3.0, 4.7, 6.2])

# Tham số khởi tạo đã cho
w_linear = 0.3
b_linear = 0.0
learning_rate_linear = 0.03
epochs_linear = 120

print("x_linear =", x_linear)
print("y_linear =", y_linear)
print("w_linear ban đầu =", w_linear)
print("b_linear ban đầu =", b_linear)
print("learning_rate_linear =", learning_rate_linear)