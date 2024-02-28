import pytorch_lightning as pl
import torch
import pandas as pd


from os.path import join as pjoin
from sklearn.metrics import r2_score
from torch.nn import  Conv2d, ConvTranspose2d,  HuberLoss,  MaxPool2d,  ReLU,  Sequential
from torch.optim import Adam


def double_conv(in_c, out_c):
    conv = Sequential(
        Conv2d(in_c, out_c, kernel_size = 2),
        ReLU(),
        Conv2d(out_c, out_c, kernel_size = 2),
        ReLU()
        )
    return conv

def triple_conv(in_c, mid_c, out_c):
    conv = Sequential(
        Conv2d(in_c, mid_c, kernel_size = 2),
        ReLU(),
        Conv2d(mid_c, out_c, kernel_size = 2),
        ReLU(),
        Conv2d(out_c, out_c, kernel_size = 2),
        ReLU())
    return conv


def load_points(df_path):
    points_df=pd.read_csv(df_path)
    novo_df= points_df[['col','row']].to_numpy(dtype=int)
    return novo_df

def crop_img(tensor, target_tensor):
    target_size=target_tensor.size()[2]
    tensor_size=tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta//2
    return tensor[:,:,delta:tensor_size-delta, delta:tensor_size-delta]

def crop_img_general(tensor,target_dim):
    tensor_h=tensor.size()[2]
    tensor_w=tensor.size()[3]
    delta_h = tensor_h - target_dim[0]
    delta_w = tensor_w - target_dim[1]
    delta_h = delta_h//2
    delta_w = delta_w//2
    return tensor [:,:,delta_h:tensor_h-delta_h, delta_w:tensor_w-delta_w]
    
data_points_path=''
        
    
class U_net_94x78(pl.LightningModule):
    def __init__(self,data_points,tensor_path = './tensors', plot_path = './plots'):
        super(U_net_94x78, self).__init__()  
        self.dist = HuberLoss(delta = 0.01)
        self.tensor_path = tensor_path 
        self.plot_path = plot_path
        self.data_points = load_points(data_points)
        
        self.down_conv_1=double_conv(8,64)
        self.down_conv_2=triple_conv(64,128,256)
        self.down_conv_3=double_conv(256,512)
        self.down_conv_4=double_conv(512,1024)

        self.max_pool_2x2 = MaxPool2d(kernel_size=2,stride=2)        

        self.up_trans_1 = ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=3, stride=1)
        self.up_trans_2 = ConvTranspose2d(
            in_channels=1024, out_channels=256, kernel_size=3, stride=3)
        self.up_conv_1=double_conv(512,256)
        self.up_trans_3 = ConvTranspose2d(
            in_channels=256, out_channels=64, kernel_size=4, stride=3)
        self.up_conv_2=double_conv(128,64)
        self.up_trans_4 = ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=3)
        self.up_trans_5 = ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.out = Conv2d(64, 1, kernel_size = 1)
        
        
    def forward(self, x):
            x1=self.down_conv_1(x)
            x2=self.max_pool_2x2(x1)
            x3=self.down_conv_2(x2)
            x4=self.max_pool_2x2(x3)
            x5=self.down_conv_3(x4)
            x6=self.max_pool_2x2(x5)
            x7=self.down_conv_4(x6)
            x8=self.up_trans_1(x7)
            x9=torch.cat([x8,x6],1)
            x10=self.up_trans_2(x9)
            x11=torch.cat([x10,x4],1)
            x12=self.up_conv_1(x11)
            x13=self.up_trans_3(x12)
            x14=torch.cat([x13,x2],1)
            x15= self.up_conv_2(x14)
            x16= self.up_trans_4(x15)
            x17=self.up_trans_5(x16)
            x18=crop_img_general(x17,[78,94])
            y = self.out(x18)

            return y
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr = 1e-03)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_true = y
        y_pred = self(x)
        
        true=torch.zeros(78,94)
        pred=torch.zeros(78,94)
       
        
        for a in self.data_points:
            true[a[1],a[0]]=y_true[0,0,a[1],a[0]] # invertir para o caso v3
            pred[a[1],a[0]]=y_pred[0,0,a[1],a[0]] # invertir para o caso v3
        

        loss = self.dist(y_true, y_pred)
        self.log('train_loss', loss, prog_bar = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_true = y
        y_pred = self(x)
        
        true=torch.zeros(78,94)
        pred=torch.zeros(78,94)
       
        
        for a in self.data_points:
            true[a[1],a[0]]=y_true[0,0,a[1],a[0]] # invertir para o caso v3
            pred[a[1],a[0]]=y_pred[0,0,a[1],a[0]] # invertir para o caso v3

        loss = self.dist(y_true,y_pred)
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
    
        
        self.log('val_loss', loss, prog_bar = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_true = y
        y_pred = self(x)

        true=[]
        pred=[]
        for count, a in enumerate(self.data_points):
            true.append(y_true[0,0,a[1],a[0]]) 
            pred.append(y_pred[0,0,a[1],a[0]]) 
        
        true=torch.Tensor(true)
        pred=torch.Tensor(pred)
        
        model_idx = self.trainer.datamodule.test_data.model_list[batch_idx].item()
        torch.save(y_true, pjoin(self.tensor_path, f'{model_idx:03d}_true.pt'))
        torch.save(y_pred, pjoin(self.tensor_path, f'{model_idx:03d}_pred.pt'))
    
        loss = self.dist(y_true, y_pred)

        r2 = r2_score(true.cpu(), pred.cpu()) 
        self.log(f'{model_idx:03d}', {'r2_score': r2, 'Loss': loss})
        return loss
    
    
class U_net_46x46(pl.LightningModule):
    def __init__(self,data_points=[],tensor_path = './tensors', plot_path = './plots'):
        super(U_net_46x46, self).__init__()  
        self.dist =  HuberLoss(delta = 0.01) #MSELoss()
        self.tensor_path = tensor_path 
        self.plot_path = plot_path
        self.data_points = load_points(data_points)
        
        self.down_conv_1=double_conv(8,64) # mudar para a quantidade de mapas de entrada
        self.down_conv_2=triple_conv(64,128,256)
        self.down_conv_3=double_conv(256,512)
        self.down_conv_4=double_conv(512,1024)

        self.max_pool_2x2 = MaxPool2d(kernel_size=2,stride=2)        

        self.up_trans_1 = ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=3, stride=1)
        self.up_trans_2 = ConvTranspose2d(
            in_channels=1024, out_channels=256, kernel_size=3, stride=3)
        self.up_conv_1=double_conv(512,256)
        self.up_trans_3 = ConvTranspose2d(
            in_channels=256, out_channels=64, kernel_size=4, stride=3)
        self.up_conv_2=double_conv(128,64)
        self.up_trans_4 = ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=3)
        self.out = Conv2d(64, 1, kernel_size = 1)
        
        
    def forward(self, x):
            x1=self.down_conv_1(x)
            x2=self.max_pool_2x2(x1)
            x3=self.down_conv_2(x2)
            x4=self.max_pool_2x2(x3)
            x5=self.down_conv_3(x4)
            x6=self.max_pool_2x2(x5)
            x7=self.down_conv_4(x6)
            x8=self.up_trans_1(x7)
            x9=torch.cat([x8,x6],1)
            x10=self.up_trans_2(x9)
            x11=torch.cat([x10,x4],1)
            x12=self.up_conv_1(x11)
            x13=self.up_trans_3(x12)
            x14=torch.cat([x13,x2],1)
            x15= self.up_conv_2(x14)
            x16= self.up_trans_4(x15)
            x17 = crop_img(x16, x)
            y = self.out(x17)

            return y
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr = 1e-03)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_true = y
        y_pred = self(x)
        
        true=torch.zeros(46,46)
        pred=torch.zeros(46,46)
       
        for a in self.data_points:
            true[a[0],a[1]]=y_true[0,0,a[0],a[1]]
            pred[a[0],a[1]]=y_pred[0,0,a[0],a[1]]
        
        
        loss=self.dist(true,pred)
        
        loss = self.dist(y_true, y_pred)
        self.log('train_loss', loss, prog_bar = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_true = y
        y_pred = self(x)
        
        true=torch.zeros(46,46)
        pred=torch.zeros(46,46)
       
        for a in self.data_points:
            true[a[0],a[1]]=y_true[0,0,a[0],a[1]]
            pred[a[0],a[1]]=y_pred[0,0,a[0],a[1]]
        
        loss = self.dist(true,pred)
        
        loss = self.dist(y_true,y_pred)
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
            
        
        self.log('val_loss', loss, prog_bar = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_true = y
        y_pred = self(x)
        
        true=[]
        pred=[]
        for count, a in enumerate(self.data_points):
            true.append(y_true[0,0,a[0],a[1]])
            pred.append(y_pred[0,0,a[0],a[1]])
        
        true=torch.Tensor(true)
        pred=torch.Tensor(pred)
        
        model_idx = self.trainer.datamodule.test_data.model_list[batch_idx].item()
        torch.save(y_true, pjoin(self.tensor_path, f'{model_idx:03d}_true.pt'))
        torch.save(y_pred, pjoin(self.tensor_path, f'{model_idx:03d}_pred.pt'))
    
        loss = self.dist(true, pred)
        loss = self.dist(y_true, y_pred)
     
        r2 = r2_score(true.cpu(), pred.cpu())
        self.log(f'{model_idx:03d}', {'r2_score': r2, 'Loss': loss})
        return loss
