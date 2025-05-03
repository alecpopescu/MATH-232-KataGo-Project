from catalyst import dl, metrics, utils

class CustomRunner(dl.Runner):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.valid_losses = []

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in ["loss"]
        }

    def handle_batch(self, batch):
        x, y = batch
        #x = x.view(len(x), -1)
        y_pred = self.model(x).squeeze(1)
        loss = self.criterion(y_pred, y)
        self.batch_metrics.update({"loss": loss})
        
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
        
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
    def on_loader_end(self, runner):
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        if self.loader_key == "train":
            self.train_losses.append(self.loader_metrics["loss"])
        elif self.loader_key == "valid":
            self.valid_losses.append(self.loader_metrics["loss"])
        super().on_loader_end(runner)
    
    