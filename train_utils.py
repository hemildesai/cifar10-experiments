import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.9, T=20):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def train_epoch(epoch, ds, model, criterion, optimizer, batch_scheduler, teacher=None):
  model.train()
  if teacher:
    teacher.eval()

  train_loss = 0
  total = 0
  correct = 0

  for i, (inputs, targets) in enumerate(ds):
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    if teacher:
      teacher_out = teacher(inputs)
      loss = loss_fn_kd(outputs, targets, teacher_out)
    else:
      loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    if batch_scheduler is not None:
        batch_scheduler.step()

  train_loss = train_loss / len(ds)
  accuracy = correct/total

  print(f'Epoch {epoch}: Loss: {train_loss} Batch Accuracy: {accuracy}')
  return {'train_loss': train_loss, 'train_accuracy': accuracy}

def evaluate(ds, model, criterion):
  model.eval()
  test_loss = 0
  correct = 0
  total = 0

  with torch.no_grad():
    for i, (inputs, targets) in enumerate(ds):
      inputs = inputs.to(device)
      targets = targets.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, targets)

      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

    test_loss = test_loss/len(ds)
    accuracy = correct/total

    print(f'Test Set evaluation: Loss: {test_loss} Test Accuracy: {accuracy}')
    return {'test_loss': test_loss, 'test_accuracy': accuracy}

def train(train_ds, test_ds, optimizer,  model, scheduler=None, epochs=24, batch_scheduler=None, swa_model=None):
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  history = []
  
  for i in tqdm.tqdm(range(epochs)):
    train_stats = train_epoch(i, train_ds, model, criterion, optimizer, batch_scheduler)
    test_stats = evaluate(test_ds, model, criterion)
    if scheduler is not None:
      scheduler.step()
    if swa_model is not None:
      swa_model.update_parameters(model)
      
    history.append((train_stats, test_stats, optimizer.param_groups[0]['lr']))
    
  return history

def distill(student, teacher, train_ds, test_ds, optimizer, scheduler=None, epochs=20):
  student = student.to(device)
  teacher = teacher.to(device)

  criterion = nn.CrossEntropyLoss()

  for i in tqdm.tqdm(range(epochs)):
    train_epoch(i, train_ds, student, criterion, optimizer, scheduler, teacher=teacher)
    evaluate(test_ds, student, criterion)
    if scheduler is not None:
      scheduler.step()