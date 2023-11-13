# Llama2RNN.c: A Lifelong RNN Model in C

[![zh](https://img.shields.io/badge/zh-简体中文-red.svg)](README.md)
[![en](https://img.shields.io/badge/en-English-green.svg)](README.en.md)

This is a Recurrent Neural Network (RNN) model designed to run indefinitely (lifelong) using the Llama2 weights.

- **llama2:** Various versions of the llama2 model weights can be used
- **rnn:** The attention sequence length for each token is fixed, so the computation and memory overhead will not increase. In theory, it supports infinite-length sequences and can read from and save memories to disk.
- **.c:** Can run on local devices, even on mobile platforms

Current repo forked from [llama2.c](https://github.com/karpathy/llama2.c).

Example

```md
# mode = llama2Rnn_toy20M_q80.bin, train_seq_len = 256, attention_seq_len = 32

Enter system prompt (optional):
(2023-11-03 11:28:49)  User: Can you provide examples of successful adoption and utilization of wearable technology in the fashion industry?
Assistant:  Yes, here are some examples of successful adoption and utilization of wearable technology in the fashion industry:

1. Eye Monitoring: Eye monitoring technology enables healthcare professionals to track data on patients' movements, symptoms, and treatment options. This data can be used to adjust the temperature, duration, and other factors that affect patient satisfaction, and predict which treatment works best for each individual patient.

2. Medical Loss: VR is a popular practice that enables physicians to make real-time changes in their patients’ health data. Patients can apply their VR training, allowing them to monitor their progress and adjust their care accordingly.

3. Health Information Management: Patients are more likely to use wearable devices, such as smartwatches, for medical diagnosis and monitoring. They can see their blood sugar levels, heatstroke, and other health conditions. This helps them to manage their symptoms and reduce the likelihood of their health conditions.

4. Surveillance: Patients can use wearables to monitor their health by monitoring their movements, activities, and other factors that contribute to their health. They can monitor their heart rate, breathing patterns, and other sensory systems to make more informed decisions about the location and timing of their menstrual cycle.

5. Safety and Health: Patients are best fluent in English as it provides them with real-time access to healthcare in a more accessible and convenient way. They can avoid getting stranded on their own or leave food, and can also save lives by smiling at home or visiting places in the house.

Overall, the use of technology in the fashion industry has transformed patient care. By providing advanced training and support to healthcare professionals, researchers have been able to provide a more accurate, efficient, and intuitive approach to patient care.

(2023-11-03 11:28:51)  User:

```

It can be seen that although the training length of the model is only 256 and the attention length is only 32, it can still generate longer coherent responses.

## How to use

### 1. Compilation

To compile the `llama2Rnn.c` code, you have two options:

#### 1.1 Fast Compilation without OpenMP Support

To quickly compile without OpenMP support, use the following command:

```bash
make runfast
```

#### 1.2 Compilation with OpenMP Support

To compile with OpenMP support, use the following command:

```bash
make runomp
```

### 2. Downloading the Model and Tokenizer

Download the required [tokenizer](https://drive.google.com/file/d/1KJei_OZHFXsc8vgqz7ZGu7V8Nw-TSwFm/view?usp=drive_link) and [model](https://drive.google.com/file/d/10UOsLSmLEWMfGitKTk8J-tbrL5J-4P6l/view?usp=drive_link) files. All avaiable models can be downloaded at [here](https://drive.google.com/drive/folders/1Px5IzuUY-H2I-bd0PRsvS0rCg9Vm7iC9?usp=sharing)

```bash
# (internal aws)All available models can be found at s3://lsy/llama2rnn.c/, and subsequent model updates will also be here

oss cp s3://lsy/llama2rnn.c/llama2Rnn_toy20M_q80.bin .
oss cp s3://lsy/llama2rnn.c/llama2_tokenizer.bin .
```

### 3. Running the Model

To run the Llama2RNN model indefinitely, use the following command:

```bash
./runqm llama2Rnn_toy20M_q80.bin -z llama2_tokenizer.bin -m chat
```

## Change Log

- 2023.11.03
    - Quantize code
    - Release 20M chat model

## Future Improvements

- Investigate and merge `run.cu` (CUDA)
- Add more models, such as 100M, 1B, and 7B
- (LoRA) Llama2 model fine-tuning
- Support for Chinese language
- Add training code
- Save memory to disk feature
- Support .txt document input
- Perceive physical time

## Known Bugs

- Pressing Enter in user input may cause invalid memory address access

## License

MIT
