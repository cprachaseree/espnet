from transformers import BertTokenizer, BertForMaskedLM, LineByLineTextDataset, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def main():
    text_root = "/home/prac0003/2_Modules/espnet/egs2/librispeech/asr1/myscripts/bert_exp/texts"
    train_path = f"{text_root}/lm_train100_external.txt"
    eval_path = f"{text_root}/dev_full.txt"

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir="./bert_exp/model",
        overwrite_output_dir=True,
        num_train_epochs=4,
        per_gpu_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2
    )
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128
    )
    eval_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=eval_path,
        block_size=128
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    
    trainer.save_model("./bert_exp/model")
    

if __name__=="__main__":
    main()
