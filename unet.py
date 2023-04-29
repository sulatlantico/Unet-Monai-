from pytorch_lightning import LightningModule
from monai.networks.nets import UNet
from monai.losses import GeneralizedDiceFocalLoss
from torchmetrics.classification import (BinaryF1Score, BinaryPrecision,
                                         BinaryAccuracy)
from torch.nn import Sequential, Sigmoid
from torch.optim import Adam

class Unet(LightningModule): 
    def __init__(self): 
        super(Unet,self).__init__()
        self.model = Sequential(UNet(spatial_dims=2,in_channels=1,
                              out_channels=1,channels=(14,28,56),
                               strides=(2,2,2), num_res_units=2,
                              kernel_size=3, dropout=.2),
                              Sigmoid())
        self.loss = GeneralizedDiceFocalLoss() 
        self.train_acc = BinaryAccuracy()
        self.valid_acc = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
        self.prec = BinaryPrecision()

    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),lr=1e-2)
        return optimizer
        
    def training_step(self,train_batch,batch_idx):
        x, y = train_batch['x'], train_batch['y'] 
        #forward pass
        z = self.model(x)
        loss = self.loss(z, y)
        t_acc = self.train_acc(z,y)
        f1_score = self.f1_score(z,y)
        precision = self.prec(z,y)
        self.log('Train_loss',loss, on_step=False, on_epoch = True, prog_bar = True,
                enable_graph = True)
        self.log('Train_acc', t_acc, on_step=False, on_epoch = True, prog_bar = True)
        self.log('F1_Score', f1_score, on_step=False, on_epoch = True, prog_bar = True)
        self.log('Precision', precision, on_step=False, on_epoch = True, prog_bar = True)
        return loss
    
    def validation_step(self,val_batch,batch_idx):
        x, y = val_batch["x"], val_batch["y"]
        #forward pass
        z = self.model(x)
        loss = self.loss(z, y)
        v_acc = self.valid_acc(z,y)
        val_f1_score = self.f1_score(z,y)
        val_precision = self.prec(z,y)
        self.log('Val_loss',loss, on_step=False, on_epoch = True, prog_bar = True)
        self.log('Valid_acc', v_acc, on_step=False, on_epoch = True, prog_bar = True)
        self.log('F1_Score', val_f1_score, on_step=False, on_epoch = True, prog_bar = True)
        self.log('Precision', val_precision, on_step=False, on_epoch = True, prog_bar = True)
        return loss
     
    def predict_step(self,batch, batch_idx,dataloader_idx=0): 
        return self(batch)  
    

if __name__ == '__main__':
    model = Unet()


