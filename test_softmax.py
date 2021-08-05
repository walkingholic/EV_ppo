import torch
import torch.nn.functional as F

torch.manual_seed(1)

# z = torch.FloatTensor([1, 2, 3])
z = torch.rand(3, 5, requires_grad=True)

print(z)
hypothesis = F.softmax(z, dim=0)
print(hypothesis)

hypothesis = F.softmax(z, dim=1)
print(hypothesis)


hypothesis = F.softmax(z, dim=-1)
print(hypothesis)

y = torch.randint(5, (3,)).long()
# y = torch.randint(3).long()
print(y)
print(y.unsqueeze(-1))
print(y)

# print(y.shape)
# print(y.size())
# print(y.dim())
#
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
print(y_one_hot)
print(y_one_hot[:, 1])


#
#
# print(y_one_hot)
#
# print(y.unsqueeze(1))


