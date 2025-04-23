from model import GovnArgs, GovnLM
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from torch import nn
from tqdm.auto import tqdm

if __name__ == '__main__':

    torch.manual_seed(54)
    device = 'cuda:0'
    args = GovnArgs(
        dim=1024,
        n_layers=36,
        n_heads=16,
        vocab_size=32000,
        feed_forward_hidden_dim=4096,
        norm_eps=1e-6,
        max_batch_size=16,
        max_seq_len=1024
    )

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", split="train")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('tokenized_smallwebtext').with_format('torch')
    dataloader = DataLoader(
        dataset['train'],
        batch_size=args.max_batch_size,
        shuffle=True
    )

    log_dir = "runs/GovnLM-0.5B"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    model = GovnLM(args).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    best_loss = float('inf')
    for step, batch in tqdm(enumerate(dataloader)):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        inputs = input_ids[:, :-1]  
        labels = labels[:, 1:]
        
        logits = model(inputs)
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, args.vocab_size),
            labels.reshape(-1)
        )
                
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        writer.add_scalar(f"loss", loss.item(), step * args.max_batch_size * args.max_seq_len)
        writer.add_scalar(f"perplexity", torch.exp(loss), step * args.max_batch_size * args.max_seq_len)
        
        if loss.item() < best_loss and step > 5000:
            best_loss = loss.item()
            torch.save(model.state_dict(), f"model_best.pt")
        if step % 5000 == 0:
            torch.save(model.state_dict(), f"model_{step}.pt")