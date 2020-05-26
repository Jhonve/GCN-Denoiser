import torch
from GCNModel import DGCNN

k_current_model = "checkpoints/23_model.t7"

dgcnn = DGCNN(8, 17, 1024, 0.5, 3)

dgcnn.load_state_dict(torch.load(k_current_model))
dgcnn.eval()

model_scirpt = torch.jit.script(dgcnn)
model_scirpt.save("script_model.pt")

print("Save success!")

dgcnn_new = torch.jit.load("script_model.pt")
dgcnn_new.cuda()
dgcnn_new.eval()

dgcnn.cuda()
inputs = torch.zeros(1, 20, 64).cuda()
a = dgcnn.forward(inputs)
print(a)

a = dgcnn_new.forward(inputs)
print(a)
