import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print(device)


# class Mymodel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mmodel = nn.Sequential(
#             nn.Linear(2, 10, bias=True),
#             nn.Tanh(),
#             nn.Linear(10, 1, bias=True),
#             nn.Tanh()
#         )
#     def forward(self, X):
#         return self.mmodel(X)
#




X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)


model = nn.Sequential(
    nn.Linear(2,2,bias=True),
    nn.ReLU(),

    nn.Linear(2,1,bias=True),
    nn.Sigmoid()
).to(device)
# model = Mymodel().to(device)

print(model)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

for step in range(10001):
    # print(model.parameters())
    optimizer.zero_grad()
    hypothesis = model(X)

    # 비용 함수
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 100 == 0: # 100번째 에포크마다 비용 출력
        print(step, cost.item())

with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())