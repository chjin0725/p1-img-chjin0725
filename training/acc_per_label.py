import numpy as np

from avgMeter import AverageMeter

class AccPerLabel():
    def __init__(self, num_class):
        self.num_class = num_class
        self.avg_meters = [0]*self.num_class
        for i in range(self.num_class):
            self.avg_meters[i] = AverageMeter()
        
        
    
    def reset(self):
        '''Reset AverageMeters.
        '''
        for i in range(self.num_class):
            self.avg_meters[i].reset()
        
    
    def update(self, pred, label):
        '''
        Args:
            pred : shape (batch_size,).
            label : shape (batch_size,).
        '''
        if not isinstance(pred, np.ndarray):
            pred = pred.numpy()
        if not isinstance(label, np.ndarray):
            label = label.numpy()
        
        true_positives = label[label==pred]
        
        for i in range(self.num_class):
            num_true_positive = (true_positives==i).sum()
            num_label_i = (label==i).sum()
            if num_label_i != 0:
                acc_i = num_true_positive / num_label_i
                self.avg_meters[i].update(acc_i, num_label_i)
            
    def show_result(self):
        temp = ['class_number num_true_positive/num_class_i = acc\n']
        for i in range(int(self.num_class/2)):
            temp.append(f'{i}   {self.avg_meters[i].sum}/{self.avg_meters[i].count}={self.avg_meters[i].avg : .2f}|| ')
        
        temp.append('\n')
        
        for i in range(int(self.num_class/2), self.num_class):
            temp.append(f'{i}   {self.avg_meters[i].sum}/{self.avg_meters[i].count}={self.avg_meters[i].avg : .2f}|| ')
        
        return ''.join(temp)