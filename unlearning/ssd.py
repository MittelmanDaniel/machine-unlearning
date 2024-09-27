import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SSD:

    def __init__(
        self,
        model,
        optimizer=None,
        selection_weight=50,
        dampening=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        is_starter=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.selection_weight = selection_weight
        self.dampening = dampening
        self.device = device
        if not optimizer:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.is_starter = is_starter

    def fisher(self, dataloader: DataLoader):
        criterion = nn.CrossEntropyLoss()
        fim = {
            name: torch.zeros_like(weight, device=weight.device)
            for name, weight in self.model.named_parameters()
        }
        for batch in dataloader:
            # if is_starter:
            #     inputs, targets = batch
            # else:
            inputs = batch["image"]
            targets = batch["age_group"]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            for name, weight in self.model.named_parameters():
                if weight.grad is not None:
                    fim[name].data += weight.grad.data ** 2

        for name, weight in self.model.named_parameters():
            fim[name].data /= len(dataloader)
        return fim

    def dampen(self, original_fim, forget_fim):
        with torch.no_grad():
            for (name, param), importance, f_importance in zip(
                self.model.named_parameters(),
                original_fim.values(),
                forget_fim.values(),
            ):
#                 print("LAYER:", name)
                noisy_selection_weight = torch.rand_like(importance) * self.selection_weight + self.selection_weight / 2
                mask = f_importance > noisy_selection_weight * importance
#                 dampening_weights = torch.ones_like(importance)
#                 dampening_weights[mask] = self.dampening * importance[mask] / f_importance[mask]
#                 dampening_weights[mask] = torch.clamp(dampening_weights[mask], max=1)
#                 noisy_dampening = torch.rand(mask.shape, min=self.dampening / 2, max=self.dampening * 2)
                noisy_dampening = torch.rand_like(importance[mask]) * self.dampening + self.dampening / 2
                dampening_weights = torch.clamp(noisy_dampening * importance[mask] / f_importance[mask], max=1)
#                 print(mask)
#                 print(dampening_weights)
                param[mask] *= dampening_weights

    def unlearn(self, full_dl: DataLoader, forget_dl: DataLoader):
        original_fim = self.fisher(full_dl)
        forget_fim = self.fisher(forget_dl)
        self.dampen(original_fim, forget_fim)
