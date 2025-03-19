import torch
import torch.nn as nn
import torch.optim as optim
import os

def train(model, train_loader, val_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = args.epochs
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()
    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    save_path = args.output_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在
    
    model.to(device)
    best_val_loss = float('inf')

    print('Start training...')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            input = torch.cat([data[:, :-1, :], label[:, :-1, -2:]], dim=-1)
            output_type = label[:, -1, 2:]  # 确保数据类型匹配 CrossEntropyLoss
            output_flow = data[:, -1, -4:-1]
            optimizer.zero_grad()
            type_out, flow_out = model(input)
            loss_cls = criterion_cls(type_out, output_type)  # 分类损失
            loss_reg = criterion_reg(flow_out, output_flow)  # 回归损失
            loss = loss_cls# + loss_reg*0.0001
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}',
              f'loss_cls:{loss_cls:.4f}',
              f'loss_reg:{loss_reg:.4f}'
              )

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                input = torch.cat([data[:, :-1, :], label[:, :-1, -2:]], dim=-1)
                output_type = label[:, -1, 2:]  # 确保数据类型匹配 CrossEntropyLoss
                output_flow = data[:, -1, -4:-1]

                type_out, flow_out = model(input)
                loss_cls = criterion_cls(type_out, output_type)  # 分类损失
                loss_reg = criterion_reg(flow_out, output_flow)  # 回归损失
                val_loss += (loss_cls ).item() #+ loss_reg*0.0001

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}',
              f'loss_cls:{loss_cls:.4f}',
              f'loss_reg:{loss_reg:.4f}'
              )

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            #设置名称，合并路径保存
            save_path_name = os.path.join(os.path.dirname(save_path), f'best_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path_name)
            print(f'Best model saved at {save_path_name} with validation loss: {best_val_loss:.4f}')
