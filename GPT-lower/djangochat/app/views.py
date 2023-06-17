from django.shortcuts import render
from django.http import JsonResponse
import json
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, GPT2LMHeadModel, BertTokenizer, \
    AutoTokenizer, AutoModel


# 定义chatglm模型的全局变量
tokenizer_glm = None
model_glm = None


# Create your views here.
def load_gpt2():
    tokenizer_gpt2 = BertTokenizer.from_pretrained('./model/trained/gpt2_LCCC/', repo_type='custom', do_lower_case=True, never_split=["[speaker1]", "[speaker2]"])
    model_gpt2 = GPT2LMHeadModel.from_pretrained('./model/trained/gpt2_LCCC/', ignore_mismatched_sizes=True)
    # 将model调为评估模式
    model_gpt2.eval()
    return tokenizer_gpt2, model_gpt2

def load_blender_bot():
    model_blender = BlenderbotForConditionalGeneration.from_pretrained('./model/trained/blenderbot-400M-distill')
    tokenizer_blender = BlenderbotTokenizer.from_pretrained('./model/trained/blenderbot-400M-distill')
    model_blender.eval()
    return tokenizer_blender, model_blender

def load_chatglm():
    global tokenizer_glm, model_glm
    tokenizer_glm = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model_glm = AutoModel.from_pretrained("model", trust_remote_code=True).half().quantize(4).cuda()
    # 设置模型推理模型
    model_glm.eval()

def to_load_chatglm(request):
    load_chatglm()
    return render(request, 'chatglm.html')

def to_chatgpt2(request):
    return render(request, 'chatgpt2.html')

def to_chatglm(request):
    return render(request, 'chatglm.html')

def to_blenderbot(request):
    return render(request, 'blenderbot.html')

def chatgpt2_view(request):
    if request.method == 'POST':
        user_input = json.loads(request.body)['user_input']
        tokenizer_gpt2, model_gpt2 = load_gpt2()
        print(tokenizer_gpt2)
        # tokenizer, model = load_gpt2()
        input_ids = tokenizer_gpt2.encode(user_input, return_tensors='pt')
        model_gpt2.to('cuda:0')
        input_ids = input_ids.to('cuda:0')
        outputs = model_gpt2.generate(input_ids,
                               do_sample=True,
                               max_length=100,
                               top_k=50,
                               top_p=0.95,
                               num_return_sequences=1)
        bot_reply = ''
        # 处理模型生成的文本
        for out in outputs:
            x = tokenizer_gpt2.decode(out)
            x = x.split('[SEP]')
            x = x[1:]
            for i in range(len(x)):
                b = x[i].split(' ')
                x[i] = ''.join(b)
            for t in x:
                if len(t) >= 3:
                    bot_reply = t
                    break

        print(bot_reply)

        # 将AI回复作为JSON响应返回给前端
        response_data = {'bot_reply': bot_reply}
        return JsonResponse(response_data)

    # 如果不是POST请求，则返回错误响应
    return JsonResponse({'error': 'Invalid request method.'}, status=400)


def blenderbot_view(request):
    tokenizer, model = load_blender_bot()
    user_input = json.loads(request.body)['user_input']

    inputs = tokenizer([user_input], return_tensors='pt')
    model.to('cuda:0')
    inputs = inputs.to('cuda:0')
    # 用解码用户输入的解码让模型生成
    reply_ids = model.generate(**inputs)
    # 解码
    bot_reply = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    print(bot_reply)

    # 将AI回复作为JSON响应返回给前端
    response_data = {'bot_reply': bot_reply}
    print(response_data)
    return JsonResponse(response_data)


history = []
def chatglm_view(request):
    # 获取已经被初始化的模型
    global tokenizer_glm, model_glm, history

    user_input = json.loads(request.body)['user_input']

    bot_reply, history = model_glm.chat(tokenizer_glm, user_input, history=history)
    print(bot_reply)
    # 清理历史对话
    if len(history) > 7:
        history = history[-7:]
    response_data = {'bot_reply': bot_reply}
    print(response_data)
    return JsonResponse(response_data)
