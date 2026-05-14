import re
import matplotlib.pyplot as plt
from datetime import datetime

# 解析日志文件的函数
def parse_log_file(path):
    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    f1_scores = []
    auc_scores = []
    reg_loss = []
    
    with open(path, 'r') as f:
        for line in f:
            # 解析epoch信息
            epoch_match = re.search(r'Epoch \[(\d+)/\d+\]', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                epochs.append(epoch)
                
            # 解析训练和验证损失
            loss_match = re.search(r'Train Loss: (\d+\.\d+), Val Loss: (\d+\.\d+)', line)
            if loss_match:
                train_loss.append(float(loss_match.group(1)))
                val_loss.append(float(loss_match.group(2)))
                
            # 解析准确率
            acc_match = re.search(r'Train Accuracy: (\d+\.\d+), Val Accuracy: (\d+\.\d+)', line)
            if acc_match:
                train_acc.append(float(acc_match.group(1)))
                val_acc.append(float(acc_match.group(2)))
                
            # 解析F1和AUC
            metric_match = re.search(r'F1: (\d+\.\d+), AUC: (\d+\.\d+)', line)
            if metric_match:
                f1_scores.append(float(metric_match.group(1)))
                auc_scores.append(float(metric_match.group(2)))
                
            # 解析回归损失
            reg_match = re.search(r'Avg Regression Loss: (\d+\.\d+)', line)
            if reg_match:
                reg_loss.append(float(reg_match.group(1)))
    
    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'f1': f1_scores,
        'auc': auc_scores,
        'reg_loss': reg_loss
    }

# 绘制图表函数
def plot_training_curves(data):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. 损失曲线
    plt.figure()
    plt.plot(data['epochs'], data['train_loss'], label='Train Loss')
    plt.plot(data['epochs'], data['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'train_val_loss_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 准确率曲线
    plt.figure()
    plt.plot(data['epochs'], data['train_acc'], label='Train Accuracy')
    plt.plot(data['epochs'], data['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'train_val_accuracy_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. F1 和 AUC 曲线
    plt.figure()
    plt.plot(data['epochs'], data['f1'], label='F1 Score')
    plt.plot(data['epochs'], data['auc'], label='AUC Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('F1 and AUC Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'f1_auc_scores_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 回归损失曲线
    plt.figure()
    plt.plot(data['epochs'], data['reg_loss'], label='Regression Loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Regression Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'regression_loss_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 主程序
if __name__ == "__main__":
    path = '/baksv/CIGIT/GXN_Liuxy/4h_base_1_200epch.log'
    log_data = parse_log_file(path)  # 替换为你的日志文件名
    plot_training_curves(log_data)