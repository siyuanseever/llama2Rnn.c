# Llama2RNN.c: A Lifelong RNN Model in C

This is an RNN (Recurrent Neural Network) model that can run forever (lifelong) using the Llama2 weights.

## 1. Compilation

To compile the `llama2Rnn.c` code, you have two options:

### 1.1 Fast Compilation without OpenMP Support

To compile quickly without OpenMP support, use the following command:

```
make runfast
```

### 1.2 Compilation with OpenMP Support

To compile with OpenMP support, use the following command:

```
make runomp
```

## 2. Downloading the Model and Tokenizer

Download the necessary model and tokenizer files using the following commands:

```
oss cp s3://lsy/llama2rnn.c/llama2Rnn_toy20M_q80.bin .
oss cp s3://lsy/llama2rnn.c/tokenizer.bin .
```

## 3. Running the Model

To run the Llama2RNN model forever, use the following command:

```
./runqm llama2Rnn_toy20M_q80.bin -z llama2_tokenizer.bin -m chat
```

by chat-gpt
