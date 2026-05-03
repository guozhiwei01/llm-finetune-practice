# -*- coding: utf-8 -*-
"""
Qwen2.5-7B SFT微调（中文Alpaca数据集）
功能：使用Unsloth框架对Qwen2.5-7B进行中文指令微调
环境：AutoDL GPU实例，建议 3090/A100
"""

# ========================================
# Step 1: 模型加载与4bit量化
# ========================================

from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


# ========================================
# Step 2: LoRA适配器配置
# ========================================

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


# ========================================
# Step 3: 中文Alpaca数据集准备
# ========================================

alpaca_prompt = """以下是一个描述任务的指令，以及一个提供更多上下文的输入。请编写一个适当完成请求的回答。

### 指令：
{}

### 输入：
{}

### 回答：
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction_zh"]
    inputs = examples["input_zh"]
    outputs = examples["output_zh"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

from datasets import load_dataset
dataset = load_dataset('silk-road/alpaca-data-gpt4-chinese', split='train',
                       cache_dir='/root/autodl-tmp/datasets')
dataset = dataset.map(formatting_prompts_func, batched=True)


# ========================================
# Step 4: SFTTrainer训练
# ========================================

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

# 显示GPU信息
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# 开始训练
trainer_stats = trainer.train()

# 训练统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")


# ========================================
# Step 5: 模型推理验证
# ========================================

FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [alpaca_prompt.format(
        "给出三个保持健康的小贴士。",
        "",
        "",
    )],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
print("\n" + "="*50)
print("推理结果:")
print("="*50)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

# 流式推理
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)

inputs = tokenizer(
    [alpaca_prompt.format(
        "请解释什么是机器学习？",
        "",
        "",
    )],
    return_tensors="pt"
).to("cuda")

print("\n" + "="*50)
print("流式推理:")
print("="*50)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256)


# ========================================
# Step 6: 模型保存
# ========================================

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("\nLoRA模型已保存到 lora_model 目录")
