import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import GST
from preprocess import SpeechDataset
from Hyperparameters import Hyperparameters as hp


def train(model, dataloader, optimizer, criterion, device):
    model.train()  # モデルを訓練モードにする
    total_loss = 0
    
    for mel_spectrogram, transcription in dataloader:
        mel_spectrogram = mel_spectrogram.to(device)  # 入力をデバイスに送る
        transcription = transcription.to(device)  # 出力のテキストもデバイスに送る
        
        optimizer.zero_grad()  # 勾配の初期化
        
        # モデルの順伝播
        style_embed = model(mel_spectrogram)
        
        # 損失関数の計算
        loss = criterion(style_embed, transcription)
        
        # 誤差逆伝播
        loss.backward()
        
        # パラメータの更新
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_loop(device, train_loader):
    model = GST().to(device)  # モデルをデバイスに転送
    criterion = torch.nn.MSELoss()  # 損失関数
    optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate)  # Adamオプティマイザー

    best_loss = float("inf")  # 最小ロスの初期値
    save_path = "best_gst_model.pth"  # モデル保存パス

    # トレーニングループ
    for epoch in range(hp.num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        
        print(f"Epoch [{epoch+1}/{hp.num_epochs}], Loss: {train_loss:.4f}")
        
        # ロスが最小の場合、モデルを保存
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

    print("Training complete.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GST().to(device)  # モデルをデバイス（GPUまたはCPU）に転送
    dataset = SpeechDataset()
    train_loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, num_workers=4)

    train_loop(device, train_loader)

if __name__ == "__main__":
    main()