from torch.autograd import Variable
import torch

class Adam_optimizer:
    def __init__(self, B1, B2, lr):
        self.B1 = B1
        self.B2 = B2
        self.lr = lr

        self.m_t = 0
        self.v_t = 0

        self.t = 1
        self.e = 1e-08

    def step_grad(self, grad, image):
        self.m_t = self.B1 * self.m_t + (1 - self.B1) * grad
        self.v_t = self.B2 * self.v_t + (1 - self.B2) * (grad ** 2)

        m_l = self.m_t / (1 - self.B1 ** self.t)
        v_l = self.v_t / (1 - self.B2 ** self.t)

        self.t += 1

        return (self.lr * m_l) / (torch.sqrt(self.v_t) + self.e)

    def step(self, grad, image):
        self.m_t = self.B1 * self.m_t + (1 - self.B1) * grad
        self.v_t = self.B2 * self.v_t + (1 - self.B2) * (grad ** 2)

        m_l = self.m_t / (1 - self.B1 ** self.t)
        v_l = self.v_t / (1 - self.B2 ** self.t)

        self.t += 1

        image = image - (self.lr * m_l) / (torch.sqrt(self.v_t) + self.e)

        return image

class Cosine_PDG_Adam:
    def __init__(self, step_size, clip_size):
        self.step_size = step_size
        self.clip_size = clip_size
        self.step_size = step_size

        self.optimizer = Adam_optimizer(lr=step_size, B1=0.9, B2=0.99)
        self.loss_function = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.step_ = 0
        
        self.mean_origin = [0.485, 0.456, 0.406]
        self.std_origin = [0.229, 0.224, 0.225]

    def step_combination(self, image_min, image_max, image, prediction, prediction_inner, target, target_inner):
        prediction_inner = prediction_inner.reshape(prediction_inner.shape[0], -1)
        target_inner = target_inner.reshape(target_inner.shape[0], -1)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        loss1 = criterion(prediction, target)
        loss2 = self.loss_function(prediction_inner, target_inner).sum()

        loss = loss1 + loss2

        grad1 = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]
        
        image = self.optimizer.step(-1 * grad1, image)
        
        image = image.detach().clone()
        image[:, 0, :, :] = image[:, 0, :, :] * self.std_origin[0] + self.mean_origin[0]
        image[:, 1, :, :] = image[:, 1, :, :] * self.std_origin[1] + self.mean_origin[1]
        image[:, 2, :, :] = image[:, 2, :, :] * self.std_origin[2] + self.mean_origin[2]
            
        image = torch.min(image, image_max)
        image = torch.max(image, image_min)
        image = image.clamp(0,1)
        
        image[:, 0, :, :] = (image[:, 0, :, :] - self.mean_origin[0]) / self.std_origin[0]
        image[:, 1, :, :] = (image[:, 1, :, :] - self.mean_origin[1]) / self.std_origin[1]
        image[:, 2, :, :] = (image[:, 2, :, :] - self.mean_origin[2]) / self.std_origin[2]

        return image

    def step(self, image_min, image_max, image, prediction, target):
        prediction = prediction.reshape(prediction.shape[0], -1)
        target = target.reshape(prediction.shape[0], -1)

        loss = (1 - self.loss_function(prediction, target)).sum()
        grad = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]
        
        image = self.optimizer.step(-1*grad, image)
        
        image[:, 0, :, :] = image[:, 0, :, :] * self.std_origin[0] + self.mean_origin[0]
        image[:, 1, :, :] = image[:, 1, :, :] * self.std_origin[1] + self.mean_origin[1]
        image[:, 2, :, :] = image[:, 2, :, :] * self.std_origin[2] + self.mean_origin[2]
            
        image = torch.min(image, image_max)
        image = torch.max(image, image_min)
        image = image.clamp(0,1)
        
        image[:, 0, :, :] = (image[:, 0, :, :] - self.mean_origin[0]) / self.std_origin[0]
        image[:, 1, :, :] = (image[:, 1, :, :] - self.mean_origin[1]) / self.std_origin[1]
        image[:, 2, :, :] = (image[:, 2, :, :] - self.mean_origin[2]) / self.std_origin[2]

        return image
    
    def reset(self):
        self.optimizer = Adam_optimizer(lr=self.step_size, B1=0.9, B2=0.99)

def model_immer_attack_auto_loss(image, model, attack, number_of_steps, device):
    model.zero_grad()
    
    input_unnorm = image.clone().detach()
        
    input_unnorm[:, 0, :, :] = input_unnorm[:, 0, :, :] * attack.std_origin[0] + attack.mean_origin[0]
    input_unnorm[:, 1, :, :] = input_unnorm[:, 1, :, :] * attack.std_origin[1] + attack.mean_origin[1]
    input_unnorm[:, 2, :, :] = input_unnorm[:, 2, :, :] * attack.std_origin[2] + attack.mean_origin[2]

    image_min = input_unnorm - attack.clip_size
    image_max = input_unnorm + attack.clip_size
    
    image_adv = image.clone().detach().to(device)
    
    image_adv.requires_grad = True
    target = model(image)
    target = (target.detach().clone().cpu()).to(device)

    for i in range(number_of_steps):
        if(i == 0):
            target_ = target + 0.00001
            prediction = model(image_adv)
            image_adv = attack.step(image_min, image_max, image_adv, prediction, target_)
            model.zero_grad()
        else:
            target_ = target
            prediction = model(image_adv)
            image_adv = attack.step(image_min, image_max, image_adv, prediction, target_)
            model.zero_grad()
    
    attack.reset()

    return image_adv


